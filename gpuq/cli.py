"""gpuq command line.

On your laptop, from your copy of the repo (it uses your usual git credentials):

    gpuq add train-1 --cmd "python train.py --epochs 10"
    gpuq add eval-1 --cmd "python eval.py" --depends-on train-1 --gpu CPU
    gpuq status
    gpuq logs train-1
    gpuq cancel train-1
    gpuq list
    gpuq upgrade        takes the latest gpuq into your copy, keeping gpuq.toml

On the dispatcher machine, inside the container:

    gpuq setup      step-by-step accounts and keys (docs/ONBOARDING.md)
    gpuq doctor     checks every piece and says what to fix

Without installing anything, `python -m gpuq ...` works the same from the root of the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import gitops, jobs, settings as settings_mod

QUEUE = "queue.json"


def find_root(start=None) -> Path:
    p = Path(start or os.getcwd()).resolve()
    for d in (p, *p.parents):
        if (d / settings_mod.FILE).exists():
            return d
    sys.exit(f"no {settings_mod.FILE} here or in a parent folder. Run gpuq from your copy of the repo")


def ops_repo(root: Path, s: dict) -> str:
    """The ops repo as git sees it: the origin of this clone when it is the ops repo, so git uses the
    same credentials as for your pushes."""
    configured = s["github"]["ops_repo"]
    try:
        url = subprocess.run(["git", "-C", str(root), "remote", "get-url", "origin"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
    except Exception:
        url = ""
    if url and configured and re.search(re.escape(configured) + r"(\.git)?/?$", url):
        return url
    return configured


def branch(root: Path, s: dict, name: str) -> gitops.Branch:
    repo = ops_repo(root, s)
    cache = Path(os.environ.get("GPUQ_CACHE") or Path.home() / ".cache" / "gpuq")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", s["github"]["ops_repo"])
    token = os.environ.get("GPUQ_GITHUB_TOKEN")
    return gitops.Branch(repo, name, cache / slug / name, token, author=_author(root), backoff_s=1.0)


def _author(root: Path) -> tuple:
    def cfg(key):
        try:
            return subprocess.run(["git", "-C", str(root), "config", key], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return ""
    return (cfg("user.name") or "gpuq", cfg("user.email") or "gpuq@users.noreply.github.com")


def read_queue(b: gitops.Branch) -> dict:
    raw = b.read(QUEUE)
    return json.loads(raw.decode("utf-8")) if raw else {"jobs": []}


def edit_queue(b: gitops.Branch, fn, message: str) -> dict:
    """Read-modify-write of queue.json, retried on a concurrent push. fn(queue) edits in place or
    raises SystemExit with the reason."""
    result = {}

    def change(root: Path):
        p = root / QUEUE
        q = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"jobs": []}
        fn(q)
        p.write_bytes((json.dumps(q, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        result["queue"] = q

    b.update(change, message)
    return result.get("queue", {"jobs": []})


def cmd_add(args, root, s):
    if not jobs.ID_RE.match(args.id):
        sys.exit("job ids use letters, digits, '.', '_' and '-', up to 64 characters")
    entry = {"id": args.id, "cmd": args.cmd, "platform": args.platform, "depends_on": args.depends_on}
    if args.gpu:
        entry["gpu"] = args.gpu.upper()
    for key in ("repo", "ref", "max_failures", "max_interruptions", "timeout_h"):
        if getattr(args, key) is not None:
            entry[key] = getattr(args, key)
    if args.no_setup:
        entry["setup"] = ""
    elif args.setup is not None:
        entry["setup"] = args.setup

    def add(q):
        ids = {j.get("id") for j in q["jobs"]}
        if args.id in ids:
            sys.exit(f"job id {args.id} already exists. Ids are never reused: pick a new one, e.g. {args.id}-2")
        unknown = [d for d in args.depends_on if d not in ids]
        if unknown:
            sys.exit(f"unknown dependencies {unknown}")
        q["jobs"].append(entry)

    edit_queue(branch(root, s, "jobs"), add, f"queue: add {args.id}")
    print(f"added {args.id}. The dispatcher picks it up within a minute. Follow it with: gpuq status")


def cmd_cancel(args, root, s):
    def cancel(q):
        hits = [j for j in q["jobs"] if j.get("id") == args.id]
        if not hits:
            sys.exit(f"no job {args.id}")
        hits[0]["cancel"] = True

    edit_queue(branch(root, s, "jobs"), cancel, f"queue: cancel {args.id}")
    print(f"cancel requested for {args.id}. A running Colab job stops within a minute. "
          "A running Kaggle job has to be stopped on kaggle.com, the dispatcher sends the link")


def cmd_list(args, root, s):
    print(json.dumps(read_queue(branch(root, s, "jobs")), indent=2, ensure_ascii=False))


def _ago(ts) -> str:
    try:
        secs = time.time() - time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")) + time.mktime(time.gmtime(0))
    except Exception:
        return "?"
    return f"{int(secs // 60)} min ago" if secs < 7200 else f"{secs / 3600:.1f} h ago"


def cmd_status(args, root, s):
    raw = branch(root, s, "runs").read("_gpuq/status.json")
    status = json.loads(raw.decode("utf-8")) if raw else None
    queue = jobs.normalize(read_queue(branch(root, s, "jobs")), s)
    if args.json:
        print(json.dumps({"status": status, "queue": queue}, indent=2, ensure_ascii=False))
        return
    if not status:
        print("the dispatcher has not reported yet. Check it on the dispatcher machine: docker logs gpuq")
        status = {"jobs": {}, "lanes": {}}
    else:
        print(f"dispatcher {str(status.get('dispatcher_sha', '?'))[:12]}, updated {status.get('updated_at')} "
              f"({_ago(status.get('updated_at'))}), {status.get('note', '')}")
        for lane, cur in (status.get("lanes") or {}).items():
            on = (status.get("lanes_enabled") or {}).get(lane, True)
            what = "off" if not on else "idle" if not cur else \
                f"{cur['job']} attempt {cur.get('attempt', '?')}, {(time.time() - cur['since']) / 60:.0f} min"
            print(f"  {lane:7s} {what}")
    recs = status.get("jobs", {})
    rows = [(j["id"], recs.get(j["id"], {}).get("status", "cancelled" if j.get("cancel") else "pending"),
             recs.get(j["id"], {}).get("platform", "-"), str(recs.get(j["id"], {}).get("attempts", 0)),
             recs.get(j["id"], {}).get("last_outcome", "-")) for j in queue]
    if rows:
        w = max(len(r[0]) for r in rows) + 2
        print(f"\n{'JOB':{w}s}{'STATUS':11s}{'PLATFORM':10s}{'ATTEMPTS':10s}LAST")
        for r in rows:
            print(f"{r[0]:{w}s}{r[1]:11s}{r[2]:10s}{r[3]:10s}{r[4]}")
    else:
        print("\nthe queue is empty. Add a job with: gpuq add <id> --cmd \"python train.py\"")


def cmd_logs(args, root, s):
    b = branch(root, s, "runs")
    if args.attempt:
        n = args.attempt
    else:
        raw = b.read("_gpuq/status.json")
        rec = (json.loads(raw.decode("utf-8")) if raw else {}).get("jobs", {}).get(args.id, {})
        lanes = (json.loads(raw.decode("utf-8")) if raw else {}).get("lanes", {})
        running = [v for v in lanes.values() if v and v.get("job") == args.id]
        n = running[0].get("attempt") if running else rec.get("attempts")
    if not n:
        sys.exit(f"no attempts of {args.id} yet")
    data = b.read(f"{args.id}/attempt-{n:02d}.log")
    if data is None:
        sys.exit(f"no log of {args.id} attempt {n} yet. Logs are published every 10 min and at the end")
    sys.stdout.write(data.decode("utf-8", errors="replace"))


KEEP = {settings_mod.FILE}  # your files, never overwritten by an upgrade


def _git(root: Path, *args, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if check and r.returncode:
        sys.exit(f"git {args[0]} failed: {r.stderr.strip()[-500:]}")
    return r.stdout


def cmd_upgrade(args, root, s):
    """A copy made with "Use this template" has its own history, so it cannot simply merge the new
    version. This takes the files of the new version as they are, except gpuq.toml, and commits them."""
    if _git(root, "status", "--porcelain", "--untracked-files=no").strip():
        sys.exit("commit or stash your changes first, then run gpuq upgrade again")
    upstream = args.upstream or s["github"]["upstream"]
    _git(root, "fetch", "-q", upstream, args.ref)
    sha = _git(root, "rev-parse", "FETCH_HEAD").strip()
    paths = [p for p in _git(root, "ls-tree", "-r", "--name-only", "FETCH_HEAD").splitlines() if p not in KEEP]
    for i in range(0, len(paths), 100):
        _git(root, "checkout", "FETCH_HEAD", "--", *paths[i:i + 100])
    if not _git(root, "status", "--porcelain", "--untracked-files=no").strip():
        print(f"already up to date with {upstream} {args.ref} ({sha[:12]})")
        return
    print(_git(root, "diff", "--cached", "--stat").rstrip())
    _git(root, "commit", "-q", "-m", f"gpuq: upgrade to {sha[:12]} from {upstream} {args.ref}")
    print(f"\ncommitted the upgrade to {sha[:12]}. Look at it with git show, then:\n"
          "  git push\n  ./install.sh update        (on the always-on computer)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gpuq", description="Personal job queue for free Colab and Kaggle machines")
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add", help="queue a job")
    a.add_argument("id")
    a.add_argument("--cmd", required=True, help='shell command run in the root of the code repo, e.g. "python train.py"')
    a.add_argument("--repo", help="code repo owner/name (default from gpuq.toml)")
    a.add_argument("--ref", help="branch, tag or commit (default from gpuq.toml)")
    a.add_argument("--setup", help="setup command (default: pip install -r requirements.txt when it exists)")
    a.add_argument("--no-setup", action="store_true", help="skip the setup step")
    a.add_argument("--platform", default="auto", choices=("auto", "colab", "kaggle"))
    a.add_argument("--gpu", help="T4, CPU, or another accelerator your plan allows (default from gpuq.toml)")
    a.add_argument("--depends-on", nargs="*", default=[], metavar="ID")
    a.add_argument("--max-failures", type=int)
    a.add_argument("--max-interruptions", type=int)
    a.add_argument("--timeout-h", type=float)
    sub.add_parser("list", help="print the queue")
    st = sub.add_parser("status", help="what the dispatcher reports")
    st.add_argument("--json", action="store_true")
    c = sub.add_parser("cancel", help="cancel a job")
    c.add_argument("id")
    lg = sub.add_parser("logs", help="log of a job's latest (or given) attempt")
    lg.add_argument("id")
    lg.add_argument("--attempt", type=int)
    up = sub.add_parser("upgrade", help="bring the latest gpuq into your copy (keeps gpuq.toml)")
    up.add_argument("--upstream", help="repo to take the new version from (default from gpuq.toml)")
    up.add_argument("--ref", default="main")
    sp = sub.add_parser("setup", help="dispatcher machine: set up accounts and keys step by step")
    sp.add_argument("--only", nargs="*", help="run only these steps")
    sub.add_parser("doctor", help="dispatcher machine: check every piece")
    sub.add_parser("dispatcher", help="dispatcher machine: run the dispatcher (the container does this)")
    args = ap.parse_args(argv)

    if args.command == "dispatcher":
        from . import dispatcher
        return dispatcher.main()
    if args.command in ("setup", "doctor"):
        from . import setup_wizard
        root = Path(os.environ.get("GPUQ_REPO_DIR") or find_root())
        s = settings_mod.load(root / settings_mod.FILE)
        return setup_wizard.run(s, root, only=args.only) if args.command == "setup" else setup_wizard.doctor(s, root)
    root = find_root()
    s = settings_mod.load(root / settings_mod.FILE)
    problems = settings_mod.check(s, need_kaggle_user=False)
    if problems:
        sys.exit("gpuq.toml: " + "; ".join(problems))
    {"add": cmd_add, "cancel": cmd_cancel, "list": cmd_list, "status": cmd_status, "logs": cmd_logs,
     "upgrade": cmd_upgrade}[args.command](args, root, s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
