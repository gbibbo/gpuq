"""`gpuq setup` and `gpuq doctor`, run on the dispatcher machine inside the container.

setup walks through every account and key, checks each one as soon as it is entered and saves it in
the secrets folder. It can be run again at any time: a value that is already there and still works
is kept unless you choose to replace it. doctor runs the same checks without asking anything.
docs/ONBOARDING.md explains how to get each key, step by step.
"""
from __future__ import annotations

import getpass
import json
import os
import secrets as pysecrets
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from . import gitops, kaggle, notify, secrets, settings as settings_mod

STEPS = ("github", "colab", "kaggle", "storage", "huggingface", "env", "ntfy", "kaggle_dataset")
GUIDE = "docs/ONBOARDING.md"


class Io:
    """Terminal input and output, replaceable in tests."""

    def __init__(self, ask=input, ask_secret=getpass.getpass, out=print):
        self.ask, self.ask_secret, self.out = ask, ask_secret, out

    def yes(self, question: str, default: bool = True) -> bool:
        ans = self.ask(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
        return default if not ans else ans.startswith("y")

    def multiline(self, prompt: str) -> str:
        self.out(prompt + " (finish with an empty line)")
        lines = []
        while True:
            line = self.ask("")
            if not line.strip():
                break
            lines.append(line)
        return "\n".join(lines) + "\n"


def _http_json(url: str, token: str = None, auth: str = "Bearer", timeout: float = 30):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "gpuq"}
    if token:
        headers["Authorization"] = f"{auth} {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception as exc:
        return 0, str(exc)


# ------------------------------------------------------------------ checks (no questions asked)

def check_github(s: dict, token: str, write_test: bool = False, state_dir=None) -> list:
    """Problems with the GitHub token, as sentences. [] when it reads the ops and code repos (and,
    with write_test, can write to branch runs)."""
    if not token:
        return ["no GitHub token"]
    problems = []
    for key in ("ops_repo", "code_repo"):
        repo = s["github"][key]
        if not repo or "/" not in repo or "://" in repo:
            continue
        code, _ = _http_json(f"https://api.github.com/repos/{repo}", token)
        if code != 200:
            problems.append(f"the token cannot read {repo} (HTTP {code}). Give it access to that repo "
                            "with Contents read and write")
    if write_test and not problems:
        d = Path(state_dir or tempfile.mkdtemp()) / "setup_check_runs"
        try:
            gitops.Branch(s["github"]["ops_repo"], "runs", d, token, retries=2, backoff_s=1).publish(
                {"_gpuq/setup-check.txt": "gpuq setup wrote this file to check that the token can push\n"},
                message="gpuq: setup check")
        except Exception as exc:
            problems.append(f"the token cannot push to {s['github']['ops_repo']}: {secrets.redact(str(exc), [token])}")
    return problems


def check_colab(interactive: bool) -> tuple:
    """(ok, detail). Interactive mode lets the Colab CLI run its sign-in flow in this terminal."""
    if interactive:
        rc = subprocess.run(["colab", "sessions"]).returncode
        return rc == 0, "colab sessions exited with code %d" % rc
    try:
        r = subprocess.run(["colab", "sessions"], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    except Exception as exc:
        return False, repr(exc)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-400:]


def check_kaggle(token: str) -> tuple:
    if not token:
        return False, "no Kaggle token"
    try:
        r = subprocess.run(["kaggle", "quota"], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
                           env={**os.environ, "KAGGLE_API_TOKEN": token})
    except Exception as exc:
        return False, repr(exc)
    return r.returncode == 0, secrets.redact((r.stdout + r.stderr).strip()[-600:], [token])


def check_rclone(conf: str, remote: str) -> tuple:
    """Creates the folder of storage.remote and lists it with the given config."""
    if not conf:
        return False, "no rclone config"
    if not remote or ":" not in remote:
        return False, "storage.remote in gpuq.toml is empty or not remote:path"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rclone.conf"
        p.write_text(conf, encoding="utf-8")
        try:
            mk = subprocess.run(["rclone", "--config", str(p), "mkdir", remote], capture_output=True, text=True,
                                timeout=180)
            ls = subprocess.run(["rclone", "--config", str(p), "lsf", "--max-depth", "1", remote], capture_output=True,
                                text=True, timeout=180)
        except Exception as exc:
            return False, repr(exc)
    frags = secrets.fragments({"RCLONE_CONF": conf})
    if mk.returncode or ls.returncode:
        return False, secrets.redact((mk.stderr + ls.stderr).strip()[-600:], frags)
    return True, f"{remote} is reachable"


def check_hf(token: str) -> tuple:
    if not token:
        return False, "no token"
    req = urllib.request.Request("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return True, "Hugging Face user " + json.load(r).get("name", "?")
    except urllib.error.HTTPError as exc:
        return False, f"Hugging Face rejected the token (HTTP {exc.code})"
    except Exception as exc:
        return False, repr(exc)


# ------------------------------------------------------------------ the wizard

def _header(io: Io, n: int, title: str, section: str):
    io.out(f"\n=== {n}. {title}   (how to get it: {GUIDE}, section {section})")


def step_github(io, s, creds, d, state_dir):
    _header(io, 1, "GitHub token", "3")
    io.out(f"Fine-grained token with Contents read and write on {s['github']['ops_repo']}"
           + (f" and {s['github']['code_repo']}" if s['github']['code_repo'] != s['github']['ops_repo'] else ""))
    if creds.get("GITHUB_TOKEN") and not check_github(s, creds["GITHUB_TOKEN"]) and not io.yes(
            "A working token is already saved. Replace it?", default=False):
        return
    while True:
        token = io.ask_secret("Paste the token (it will not show) and press Enter: ").strip()
        if not token.startswith(("github_pat_", "ghp_")):
            io.out("That does not look like a GitHub token (they start with github_pat_ or ghp_)")
            continue
        io.out("checking read access, then a test push to branch runs ...")
        problems = check_github(s, token, write_test=True, state_dir=state_dir)
        if not problems:
            creds["GITHUB_TOKEN"] = token
            secrets.save_credentials(creds, d)
            io.out("ok, saved")
            return
        for p in problems:
            io.out("  problem: " + p)
        if not io.yes("Try another token?"):
            return


def step_colab(io, s, creds, d, state_dir):
    if not s["colab"]["enabled"]:
        return
    _header(io, 2, "Colab sign-in", "4")
    ok, _ = check_colab(interactive=False)
    if ok and not io.yes("The Colab CLI is already signed in. Sign in again?", default=False):
        return
    io.out("The Colab CLI prints a link. Open it in any browser, sign in with the Google account you use\n"
           "for Colab, approve, and paste the code Google shows you back here.")
    ok, detail = check_colab(interactive=True)
    io.out("ok, signed in" if ok else f"  problem: {detail}. Run gpuq setup --only colab to try again")


def step_kaggle(io, s, creds, d, state_dir):
    if not s["kaggle"]["enabled"]:
        return
    _header(io, 3, "Kaggle API token", "5")
    current = secrets.read(secrets.KAGGLE_TOKEN, d)
    if current and check_kaggle(current)[0] and not io.yes("A working Kaggle token is saved. Replace it?", default=False):
        return
    while True:
        token = io.ask_secret("Paste the Kaggle API token (it will not show) and press Enter: ").strip()
        ok, detail = check_kaggle(token)
        if ok:
            secrets.write(secrets.KAGGLE_TOKEN, token, d)
            io.out("ok, saved. Your GPU quota:\n" + detail)
            return
        io.out(f"  problem: {detail}")
        if not io.yes("Try another token?"):
            return


def step_storage(io, s, creds, d, state_dir):
    _header(io, 4, "Storage for checkpoints and results (rclone)", "6")
    remote = s["storage"]["remote"]
    if not remote:
        io.out("storage.remote is empty in gpuq.toml, so jobs get no remote. Skipping")
        return
    if creds.get("RCLONE_CONF") and check_rclone(creds["RCLONE_CONF"], remote)[0] and not io.yes(
            "A working rclone config is saved. Replace it?", default=False):
        return
    while True:
        conf = io.multiline("Paste the section of your laptop's rclone.conf for this remote, starting with [name]")
        if "[" not in conf or "type" not in conf:
            io.out("That does not look like an rclone config section")
        else:
            io.out(f"checking {remote} ...")
            ok, detail = check_rclone(conf, remote)
            if ok:
                creds["RCLONE_CONF"] = conf
                secrets.save_credentials(creds, d)
                io.out(f"ok, saved. {detail}")
                return
            io.out(f"  problem: {detail}")
        if not io.yes("Try again?"):
            return


def step_huggingface(io, s, creds, d, state_dir):
    _header(io, 5, "Hugging Face token (optional, makes model downloads faster)", "7")
    if creds.get("HF_TOKEN") and check_hf(creds["HF_TOKEN"])[0] and not io.yes("A working token is saved. Replace it?",
                                                                                 default=False):
        return
    if not creds.get("HF_TOKEN") and not io.yes("Add a Hugging Face token?", default=False):
        return
    token = io.ask_secret("Paste the token (it will not show) and press Enter, or just Enter to skip: ").strip()
    if not token:
        return
    ok, detail = check_hf(token)
    if ok:
        creds["HF_TOKEN"] = token
        secrets.save_credentials(creds, d)
    io.out(("ok, saved. " if ok else "  problem: ") + detail)


def step_env(io, s, creds, d, state_dir):
    _header(io, 6, "Other secrets your jobs need (optional, e.g. WANDB_API_KEY)", "8")
    env = dict(creds.get("env") or {})
    if env:
        io.out("saved now: " + ", ".join(sorted(env)))
    while io.yes("Add, replace or remove one?", default=False):
        name = io.ask("Variable name: ").strip()
        if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
            io.out("names use letters, digits and _, and do not start with a digit")
            continue
        value = io.ask_secret(f"Value of {name} (it will not show, empty to remove it): ").strip()
        if value:
            env[name] = value
        elif env.pop(name, None) is not None:
            io.out(f"removed {name}")
    creds["env"] = env
    secrets.save_credentials(creds, d)


def step_ntfy(io, s, creds, d, state_dir):
    _header(io, 7, "Phone notifications with ntfy (optional)", "9")
    topic = secrets.read(secrets.NTFY_TOPIC, d)
    if topic and not io.yes("Notifications are set up. Make a new topic?", default=False):
        return
    if not topic and not io.yes("Set up notifications?"):
        return
    topic = "gpuq-" + pysecrets.token_urlsafe(18).replace("-", "x").replace("_", "y")
    secrets.write(secrets.NTFY_TOPIC, topic, d)
    io.out(f"Your private topic is\n\n    {topic}\n\nIn the ntfy app tap + and subscribe to it. Anyone who knows it can "
           "read your notifications, so do not share it.")
    sent = notify.send(topic, "gpuq", "Test notification from gpuq setup")
    io.out("a test notification was sent" if sent else "  problem: could not reach ntfy.sh")


def step_kaggle_dataset(io, s, creds, d, state_dir):
    if not s["kaggle"]["enabled"]:
        return
    _header(io, 8, "Private Kaggle dataset with the credentials", "5")
    token = secrets.read(secrets.KAGGLE_TOKEN, d)
    if not token or not s["kaggle"]["username"]:
        io.out("needs the Kaggle token and kaggle.username in gpuq.toml. Skipping")
        return
    io.out(f"Kaggle runs read your credentials from the private dataset {s['kaggle']['username']}/"
           f"{s['kaggle']['dataset']}. Never make it public. Publishing ...")
    rc, out = kaggle.publish_credentials(secrets.load_credentials(d), s["kaggle"]["username"], s["kaggle"]["dataset"],
                                         env={**os.environ, "KAGGLE_API_TOKEN": token})
    io.out(("ok. " if rc == 0 else "  problem: ") + out[-600:])


STEP_FUNCS = {"github": step_github, "colab": step_colab, "kaggle": step_kaggle, "storage": step_storage,
              "huggingface": step_huggingface, "env": step_env, "ntfy": step_ntfy,
              "kaggle_dataset": step_kaggle_dataset}


def run(s: dict, root: Path, only=None, io: Io = None, secrets_dir=None, state_dir=None) -> int:
    io = io or Io()
    problems = settings_mod.check(s)
    if problems:
        for p in problems:
            io.out("gpuq.toml: " + p)
        io.out("Fix gpuq.toml in your repo (and push it), then run gpuq setup again")
        return 2
    d = secrets.folder(secrets_dir)
    creds = secrets.load_credentials(d)
    unknown = [x for x in (only or []) if x not in STEPS]
    if unknown:
        io.out(f"unknown steps {unknown}. Steps: {', '.join(STEPS)}")
        return 2
    io.out(f"gpuq setup for {s['github']['ops_repo']}. Secrets are saved in {d} on this machine only.")
    for name in only or STEPS:
        STEP_FUNCS[name](io, s, creds, d, state_dir)
        creds = secrets.load_credentials(d)
        if name == "storage" and creds.get("RCLONE_CONF") and s["kaggle"]["enabled"] and "kaggle_dataset" not in (only or STEPS):
            io.out("the rclone config changed: run gpuq setup --only kaggle_dataset so Kaggle gets it too")
    io.out("\nDone. Check everything with: gpuq doctor")
    return 0


def doctor(s: dict, root: Path, out=print, secrets_dir=None) -> int:
    d = secrets.folder(secrets_dir)
    creds = secrets.load_credentials(d)
    results = []

    def item(name, ok, detail=""):
        results.append(ok)
        out(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))

    problems = settings_mod.check(s)
    item("gpuq.toml", not problems, "; ".join(problems))
    gh = check_github(s, creds.get("GITHUB_TOKEN"))
    item("GitHub token", not gh, "; ".join(gh))
    if s["colab"]["enabled"]:
        ok, detail = check_colab(interactive=False)
        item("Colab sign-in", ok, detail + ". Run: docker exec -it gpuq colab sessions")
    if s["kaggle"]["enabled"]:
        ok, detail = check_kaggle(secrets.read(secrets.KAGGLE_TOKEN, d))
        item("Kaggle token", ok, detail)
    if s["storage"]["remote"]:
        ok, detail = check_rclone(creds.get("RCLONE_CONF"), s["storage"]["remote"])
        item("rclone storage", ok, detail)
    if creds.get("HF_TOKEN"):
        ok, detail = check_hf(creds["HF_TOKEN"])
        item("Hugging Face token", ok, detail)
    item("ntfy topic", bool(secrets.read(secrets.NTFY_TOPIC, d)), "not set, you will get no notifications")
    return 0 if all(results) else 1
