"""Stands in for the Kaggle CLI in the tests. A pushed kernel runs the notebook's cell with this Python,
detached, with its output in FAKE_STATE/kaggle/log.txt.

    FAKE_STATE                  folder for this fake's bookkeeping
    FAKE_KAGGLE_NO_QUOTA=1      push fails as when the weekly GPU quota is spent
    FAKE_KAGGLE_CUT=N           the N-th run dies before printing anything (status ERROR)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

state = Path(os.environ["FAKE_STATE"]) / "kaggle"
state.mkdir(parents=True, exist_ok=True)
book = state / "book.json"
data = json.loads(book.read_text()) if book.exists() else {"runs": 0, "calls": []}
args = sys.argv[1:]
data["calls"].append(" ".join(args[:2]))


def save():
    book.write_text(json.dumps(data))


RUNNER = r'''
import json, subprocess, sys
p = json.loads(sys.argv[1])
with open(p["log"], "w", encoding="utf-8") as log:
    rc = 1 if p["cut"] else subprocess.call([sys.executable, p["script"]], stdout=log, stderr=subprocess.STDOUT)
open(p["rc"], "w").write(str(rc))
'''

if args[:2] == ["kernels", "push"]:
    folder = Path(args[args.index("-p") + 1])
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    if meta["enable_gpu"] and os.environ.get("FAKE_KAGGLE_NO_QUOTA"):
        save()
        print("409 Conflict: Maximum weekly GPU quota of 30.00 hours reached.")
        sys.exit(1)
    nb = json.loads((folder / meta["code_file"]).read_text(encoding="utf-8"))
    data["runs"] += 1
    data["last_meta"] = meta
    save()
    script = state / "run.py"
    script.write_text(nb["cells"][0]["source"], encoding="utf-8")
    for f in ("log.txt", "rc.txt"):
        (state / f).unlink(missing_ok=True)
    cut = str(data["runs"]) == os.environ.get("FAKE_KAGGLE_CUT")
    kw = ({"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
          if os.name == "nt" else {"start_new_session": True})
    subprocess.Popen([sys.executable, "-c", RUNNER, json.dumps({"log": str(state / "log.txt"), "rc": str(state / "rc.txt"),
                      "script": str(script), "cut": cut})], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, **kw)
    print(f"Kernel version {data['runs']} successfully pushed.  Please check progress at https://www.kaggle.com/code/x")
elif args[:2] == ["kernels", "status"]:
    save()
    rc = state / "rc.txt"
    status = "RUNNING" if not rc.exists() else ("COMPLETE" if rc.read_text().strip() == "0" else "ERROR")
    print(f'{args[2]} has status "KernelWorkerStatus.{status}"')
elif args[:2] == ["kernels", "output"]:
    save()
    dest = Path(args[args.index("-p") + 1])
    dest.mkdir(parents=True, exist_ok=True)
    text = (state / "log.txt").read_text(encoding="utf-8") if (state / "log.txt").exists() else ""
    (dest / "gpuq-launcher.log").write_text(json.dumps([{"stream_name": "stdout", "time": 1.0, "data": text}]))
    print("Output file downloaded")
elif args[:2] == ["kernels", "logs"]:  # the real CLI prints the session log as a JSON list
    save()
    text = (state / "log.txt").read_text(encoding="utf-8") if (state / "log.txt").exists() else ""
    print(json.dumps([{"stream_name": "stdout", "time": 1.0, "data": text}]) if text else "")
else:
    save()
    print("unknown")
    sys.exit(2)
