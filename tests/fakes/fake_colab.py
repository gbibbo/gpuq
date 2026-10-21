"""Stands in for the Colab CLI in the tests. The "VM" is the local machine: upload copies a file to the
given path, exec runs the script with this Python. Behaviour switches come from environment variables.

    FAKE_STATE                 folder for this fake's bookkeeping
    FAKE_COLAB_NO_GPU=1        `new` with --gpu fails as when Colab gives no GPU
    FAKE_COLAB_LOSE_START=N    the start exec of the N-th session loses the VM before running anything
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

state = Path(os.environ["FAKE_STATE"])
state.mkdir(parents=True, exist_ok=True)
book = state / "colab.json"
data = json.loads(book.read_text()) if book.exists() else {"sessions": 0, "calls": []}


def save():
    book.write_text(json.dumps(data))


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


args = sys.argv[1:]
cmd = args[0] if args else ""
data["calls"].append(cmd)
if cmd == "new":
    if "--gpu" in args and os.environ.get("FAKE_COLAB_NO_GPU"):
        save()
        print("Error: no GPU backend is available right now")
        sys.exit(1)
    data["sessions"] += 1
    save()
    print(f"Created session {opt(args, '-s')}")
elif cmd == "upload":
    save()
    src, dst = args[-2], args[-1]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print("uploaded")
elif cmd == "exec":
    script = opt(args, "-f")
    is_start = Path(script).name == "start.py"
    if is_start and str(data["sessions"]) == os.environ.get("FAKE_COLAB_LOSE_START"):
        save()
        print("RuntimeError: Connection to remote host was lost.")
        sys.exit(1)
    save()
    env = dict(os.environ)
    if "--env" in args:
        k, _, v = opt(args, "--env").partition("=")
        env[k] = v
    sys.exit(subprocess.run([sys.executable, script], env=env).returncode)
elif cmd in ("stop", "sessions"):
    save()
    print("ok")
else:
    save()
    print(f"unknown command {cmd}")
    sys.exit(2)
