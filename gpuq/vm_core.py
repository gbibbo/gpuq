"""Code that runs on the Colab or Kaggle VM. Standard library only.

The dispatcher embeds this whole file in the script it sends to the VM (see vm.py) and appends a
call to main() with the job's parameters. main() loads the credentials, installs rclone when a
config was given, checks out the job's code and this gpuq copy (for the helpers in gpuq.storage),
runs the job's setup command and then the job's command. On Colab the command runs detached from
the kernel and the dispatcher polls its log. On Kaggle it runs in the foreground.

Every line this file prints starts with [gpuq]. The four markers below tell the dispatcher how an
attempt ended. Secrets are never printed, only the names of the variables they go into.
"""
import base64
import glob
import json
import os
import platform as _platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

MARK_SETUP_FAILED = "[gpuq] setup failed"
MARK_STARTED = "[gpuq] job started"
MARK_DONE = "[gpuq] job finished"
MARK_EXIT = "[gpuq] job exited with code"
RCLONE_VERSION = "1.75.2"


class SetupFailed(Exception):
    pass


def say(msg):
    print(f"[gpuq] {msg}", flush=True)


def load_credentials(p):
    """The bundle written by `gpuq setup`. On Colab it was uploaded for this attempt and is deleted
    after reading. On Kaggle it comes from the private credentials dataset attached to the kernel."""
    path = p.get("creds_path")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            creds = json.load(fh)
        if p.get("delete_creds"):
            os.remove(path)
        return creds
    for pattern in p.get("creds_globs") or []:
        for found in sorted(glob.glob(pattern, recursive=True)):
            with open(found, encoding="utf-8") as fh:
                return json.load(fh)
    raise SetupFailed("no credentials found. On Kaggle the private credentials dataset must exist "
                      "(gpuq setup publishes it) and the account must have a verified phone")


def export_env(creds, home):
    """Puts the secrets in the environment of the job. Returns the variable names."""
    names = []
    for key in ("GITHUB_TOKEN", "HF_TOKEN"):
        if creds.get(key):
            os.environ[key] = creds[key]
            names.append(key)
    for key, value in (creds.get("env") or {}).items():
        os.environ[str(key)] = str(value)
        names.append(str(key))
    conf = creds.get("RCLONE_CONF")
    if conf:
        if "[" not in conf:  # base64 in one line
            conf = base64.b64decode(conf).decode("utf-8")
        folder = os.path.join(home, ".config", "rclone")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "rclone.conf")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(conf)
        os.chmod(path, 0o600)
        os.environ["RCLONE_CONFIG"] = path
        names.append("RCLONE_CONFIG")
    return names


def ensure_rclone(bin_dir):
    if shutil.which("rclone"):
        return
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(
        _platform.machine().lower(), "amd64")
    name = f"rclone-v{RCLONE_VERSION}-linux-{arch}"
    target = os.path.join(bin_dir, "rclone.zip")
    os.makedirs(bin_dir, exist_ok=True)
    for url in (f"https://github.com/rclone/rclone/releases/download/v{RCLONE_VERSION}/{name}.zip",
                f"https://downloads.rclone.org/v{RCLONE_VERSION}/{name}.zip"):
        try:
            # downloads.rclone.org answers 403 to Python's default user agent
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(target, "wb") as fh:
                shutil.copyfileobj(r, fh)
            break
        except Exception as exc:
            say(f"could not download rclone from {url} ({type(exc).__name__})")
    else:
        raise SetupFailed("could not install rclone")
    with zipfile.ZipFile(target) as z:
        member = next(m for m in z.namelist() if m.endswith("/rclone"))
        with z.open(member) as src, open(os.path.join(bin_dir, "rclone"), "wb") as dst:
            shutil.copyfileobj(src, dst)
    os.chmod(os.path.join(bin_dir, "rclone"), 0o755)
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    say(f"rclone {RCLONE_VERSION} installed")


def check_internet():
    try:
        urllib.request.urlopen("https://github.com", timeout=15)
        say("internet ok")
    except Exception as exc:
        say(f"no internet ({type(exc).__name__}). On Kaggle, verify the phone number of the account")


def git_env(token):
    """Token in an HTTP header through GIT_CONFIG_* variables, never in a URL or .git/config."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"})
    return env


def repo_url(repo):
    if "://" in repo or os.path.isabs(repo) or os.path.exists(repo):
        return repo
    return f"https://github.com/{repo}.git"


def _rmtree(path):
    def writable(func, p, _exc):  # git objects are read-only on Windows
        os.chmod(p, 0o700)
        func(p)
    if os.path.exists(path):
        shutil.rmtree(path, **({"onexc": writable} if sys.version_info >= (3, 12) else {"onerror": writable}))


def checkout(repo, ref, dest, env):
    """Shallow checkout of a branch, tag or commit. Returns the commit SHA."""
    _rmtree(dest)
    os.makedirs(dest)
    for args in (["init", "-q"], ["config", "core.autocrlf", "false"], ["remote", "add", "origin", repo_url(repo)],
                 ["fetch", "-q", "--depth", "1", "origin", ref], ["checkout", "-q", "FETCH_HEAD"]):
        r = subprocess.run(["git", *args], cwd=dest, env=env, capture_output=True, text=True)
        if r.returncode:
            raise SetupFailed(f"git {args[0]} failed for {repo} at {ref}: {r.stderr.strip()[-400:]}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, env=env, capture_output=True,
                          text=True).stdout.strip()


def _shell():
    return None if os.name == "nt" else (shutil.which("bash") or "/bin/sh")


def stream(cmd, cwd, env):
    """Runs a shell command and prints its output line by line. Notebook kernels do not show the
    output of child processes unless it passes through Python."""
    proc = subprocess.Popen(cmd, shell=True, executable=_shell(), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def run_setup(setup, cwd, env):
    if setup is None:
        if not os.path.exists(os.path.join(cwd, "requirements.txt")):
            say("no setup command and no requirements.txt, nothing to install")
            return
        setup = f'"{sys.executable}" -m pip install -q -r requirements.txt'
    if not setup.strip():
        return
    say(f"setup: {setup}")
    t0 = time.time()
    rc = stream(setup, cwd, env)
    if rc:
        raise SetupFailed(f"setup command exited with code {rc}")
    say(f"setup done in {time.time() - t0:.0f} s")


def job_env(p, code_dir, ops_dir, sha):
    env = {**os.environ, "GPUQ_JOB_ID": p["job_id"], "GPUQ_ATTEMPT": str(p["attempt"]),
           "GPUQ_PLATFORM": p["platform"], "GPUQ_CODE_DIR": code_dir, "GPUQ_CODE_SHA": sha,
           "GPUQ_OPS_REPO": p.get("ops_repo") or "", "GPUQ_SCRATCH": p["scratch"], "PYTHONUNBUFFERED": "1",
           "PYTHONIOENCODING": "utf-8"}
    if p.get("remote"):
        env["GPUQ_REMOTE"] = p["remote"]
    if ops_dir:
        env["PYTHONPATH"] = ops_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


WRAPPER = r'''
import json, os, shutil, subprocess, sys
p = json.loads(sys.argv[1])
shell = None if os.name == "nt" else (shutil.which("bash") or "/bin/sh")
with open(p["log"], "ab") as log:
    rc = subprocess.call(p["cmd"], shell=True, executable=shell, cwd=p["cwd"], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT)
with open(p["rc"] + ".part", "w") as fh:
    fh.write(str(rc))
os.replace(p["rc"] + ".part", p["rc"])
'''


def start_detached(cmd, cwd, env, log, rc):
    """The job outlives the exec that started it. Its output goes to `log` and its exit code to `rc`."""
    for f in (log, rc, rc + ".part"):
        if os.path.exists(f):
            os.remove(f)
    kw = ({"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
          if os.name == "nt" else {"start_new_session": True})
    proc = subprocess.Popen([sys.executable, "-c", WRAPPER, json.dumps({"cmd": cmd, "cwd": cwd, "log": log, "rc": rc})],
                            env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, **kw)
    print(f"{MARK_STARTED}, pid {proc.pid}", flush=True)


def main(p):
    """p is the dict built by vm.params()."""
    say(f"job {p['job_id']} attempt {p['attempt']} on {p['platform']}")
    home = p.get("home") or os.path.expanduser("~")
    code_dir, ops_dir = os.path.join(p["workdir"], "code"), None
    try:
        names = export_env(load_credentials(p), home)
        say("credentials loaded into " + (", ".join(sorted(names)) or "nothing"))
        if os.environ.get("RCLONE_CONFIG") and p.get("install_rclone", True):
            ensure_rclone(os.path.join(home, ".local", "bin"))
        if p.get("check_internet", True):
            check_internet()
        genv = git_env(os.environ.get("GITHUB_TOKEN"))
        sha = checkout(p["repo"], p["ref"], code_dir, genv)
        say(f"code {p['repo']} at {p['ref']} is {sha[:12]}")
        if p.get("ops_repo") and p.get("ops_sha"):
            ops_dir = os.path.join(p["workdir"], "gpuq")
            checkout(p["ops_repo"], p["ops_sha"], ops_dir, genv)
        os.makedirs(p["scratch"], exist_ok=True)
        env = job_env(p, code_dir, ops_dir, sha)
        run_setup(p.get("setup"), code_dir, env)
    except SetupFailed as exc:
        print(f"{MARK_SETUP_FAILED}: {exc}", flush=True)
        raise SystemExit(1)
    except Exception as exc:  # a bug or a surprise of the platform: report it, do not count it as a cut
        print(f"{MARK_SETUP_FAILED}: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
    say(f"running: {p['cmd']}")
    if p["mode"] == "detached":
        start_detached(p["cmd"], code_dir, env, p["log"], p["rc"])
        return
    rc = stream(p["cmd"], code_dir, env)
    if rc == 0:
        print(MARK_DONE, flush=True)
        return
    print(f"{MARK_EXIT} {rc}", flush=True)
    raise SystemExit(rc)
