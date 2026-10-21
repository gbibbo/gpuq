"""Builds what the dispatcher sends to a VM: the start script, the Colab poll script and the Kaggle
notebook.

Colab. A VM whose only traffic is one long `colab exec` is reclaimed after about 12 minutes and the
CLI then waits forever. So the job does not live inside an exec. The start script (vm_core.main in
detached mode) sets everything up, starts the job detached from the kernel and returns. Every minute
the dispatcher runs the poll script, a short exec that brings back the new part of the job's log,
free memory, free disk, GPU use and, at the end, the exit code.

Kaggle. A kernel pushed with the Kaggle CLI runs to the end by itself, so the notebook runs
vm_core.main in the foreground and the dispatcher follows `kaggle kernels status`.
"""
from __future__ import annotations

import json
from pathlib import Path

CORE = (Path(__file__).with_name("vm_core.py")).read_text(encoding="utf-8")
BEGIN = "__GPUQ_BEGIN__"
MARK = "__GPUQ__"
CHUNK = 100_000

COLAB_ROOT = "/content"
KAGGLE_WORKING = "/kaggle/working"
KAGGLE_INPUT = "/kaggle/input"
CREDENTIALS_FILE = "gpuq_credentials.json"


def colab_paths(root: str = COLAB_ROOT) -> dict:
    return {"creds_path": f"{root}/{CREDENTIALS_FILE}", "workdir": f"{root}/gpuq", "scratch": f"{root}/scratch",
            "log": f"{root}/gpuq_job.log", "rc": f"{root}/gpuq_job.rc"}


def kaggle_paths(working: str = KAGGLE_WORKING, inputs: str = KAGGLE_INPUT, scratch: str = "/tmp/gpuq-scratch") -> dict:
    return {"creds_globs": [f"{inputs}/**/{CREDENTIALS_FILE}"], "workdir": f"{working}/gpuq", "scratch": scratch}


def params(job: dict, attempt: int, platform: str, ops_repo: str, ops_sha: str, remote: str, paths: dict,
           **extra) -> dict:
    """Everything vm_core.main needs for one attempt. No secrets."""
    p = {"job_id": job["id"], "attempt": attempt, "platform": platform, "repo": job["repo"], "ref": job["ref"],
         "setup": job.get("setup"), "cmd": job["cmd"], "ops_repo": ops_repo, "ops_sha": ops_sha, "remote": remote,
         "mode": "detached" if platform == "colab" else "foreground", "delete_creds": platform == "colab", **paths}
    p.update(extra)
    return p


def start_script(p: dict) -> str:
    return CORE + f"\n\nmain(json.loads({json.dumps(p)!r}))\n"


POLL = '''# gpuq poll: the new part of the job log from byte GPUQ_OFFSET, then one status line
import os as _os, shutil as _shutil, subprocess as _sp
_off = int(_os.environ.get("GPUQ_OFFSET") or 0)
_rc = None
if _os.path.exists(@RC@):  # read before the log: once the code exists the log is complete
    with open(@RC@) as _fh:
        _rc = _fh.read().strip() or "-1"
_raw = b""
if _os.path.exists(@LOG@):
    with open(@LOG@, "rb") as _fh:
        _fh.seek(_off)
        _raw = _fh.read(@CHUNK@)
_more = len(_raw) == @CHUNK@
_done = _rc is not None and not _more
if not _done:  # whole lines only while the job is still writing
    _raw = _raw[:_raw.rindex(b"\\n") + 1] if b"\\n" in _raw else (_raw if _more else b"")
try:
    with open("/proc/meminfo") as _fh:
        _mem = {_l.split(":")[0]: int(_l.split()[1]) // 1024 for _l in _fh}
except OSError:
    _mem = {}
_disk = _shutil.disk_usage(_os.path.dirname(@LOG@) or ".").free / 2**30
_gpu = "-"
if _shutil.which("nvidia-smi"):
    try:
        _q = _sp.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        _gpu = ";".join(_l.replace(" ", "") for _l in _q) or "-"  # util%,MiB per GPU
    except Exception:
        pass
print("@BEGIN@")
print(_raw.decode("utf-8", "replace"), end="")
print(f"\\n@MARK@ {_off + len(_raw)} {'exit ' + _rc if _done else 'running'} "
      f"{_mem.get('MemAvailable', -1)} {_mem.get('MemTotal', -1)} {_disk:.1f} {_gpu}")
'''


def poll_script(log: str, rc: str) -> str:
    out = POLL
    for key, value in (("@LOG@", repr(log)), ("@RC@", repr(rc)), ("@CHUNK@", str(CHUNK)), ("@BEGIN@", BEGIN),
                       ("@MARK@", MARK)):
        out = out.replace(key, value)
    return out


def parse_poll(out: str):
    """Fields of one poll's output, or None when the poll did not reach the VM."""
    b = out.find(BEGIN + "\n")
    m = out.rfind("\n" + MARK + " ")
    if b < 0 or m < b:
        return None
    fields = out[m + len(MARK) + 2:].split()
    try:
        done = fields[1] == "exit"
        rest = fields[3:] if done else fields[2:]
        return {"text": out[b + len(BEGIN) + 1:m], "offset": int(fields[0]), "rc": int(fields[2]) if done else None,
                "mem_avail": int(rest[0]), "mem_total": int(rest[1]), "disk_gb": float(rest[2]),
                "gpu": rest[3] if len(rest) > 3 else "-"}
    except (IndexError, ValueError):
        return None


def kaggle_notebook(script: str) -> str:
    """A one-cell notebook. Kaggle runs it top to bottom when the kernel is pushed."""
    nb = {"cells": [{"cell_type": "code", "id": "gpuq", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": script}],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1)
