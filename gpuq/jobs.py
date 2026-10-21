"""The queue and the decisions of the dispatcher. Pure functions, no network.

A queue entry (branch jobs, file queue.json, edited with `gpuq add` and `gpuq cancel`):

    {"id": "train-1", "cmd": "python train.py --epochs 10", "repo": "me/research", "ref": "main",
     "setup": null, "platform": "auto", "gpu": "T4", "depends_on": [], "max_failures": 2,
     "max_interruptions": 10, "timeout_h": 12, "cancel": false}

Only id and cmd are required. setup null means `pip install -r requirements.txt` when the file exists.
gpu "CPU" asks for a machine without accelerator. platform is colab, kaggle or auto. An auto job
takes Colab when it is free, because Colab does not spend Kaggle's weekly quota, and Kaggle when
Colab is busy or gives no GPU. CPU jobs go to Kaggle unless colab.cpu_jobs is true. Once a job has
run on a platform it stays there, so a resumed job keeps the same kind of hardware.
"""
from __future__ import annotations

import re

from . import vm_core

LANES = ("colab", "kaggle")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
FINISHED = ("done", "failed", "cancelled")
OUTCOMES = ("done", "job_error", "setup_error", "interrupted", "cancelled")


def normalize(raw, settings: dict) -> list:
    """Valid entries of queue.json with defaults filled in, in queue order. Invalid ones are skipped."""
    g, d = settings["github"], settings["jobs"]
    jobs, seen = [], set()
    for j in (raw or {}).get("jobs", []):
        if not isinstance(j, dict) or not ID_RE.match(str(j.get("id", ""))) or not str(j.get("cmd") or "").strip():
            continue
        if j["id"] in seen:
            continue
        seen.add(j["id"])
        job = {"repo": g["code_repo"], "ref": g["ref"], "setup": None, "platform": "auto", "gpu": d["gpu"],
               "depends_on": [], "max_failures": d["max_failures"], "max_interruptions": d["max_interruptions"],
               "timeout_h": d["timeout_h"], "cancel": False}
        job.update({k: v for k, v in j.items() if v is not None})
        if job["platform"] not in ("auto", *LANES):
            job["platform"] = "auto"
        job["gpu"] = str(job["gpu"] or "T4").upper()
        job["depends_on"] = [str(x) for x in (job.get("depends_on") or [])]
        jobs.append(job)
    return jobs


def is_cpu(job: dict) -> bool:
    return job["gpu"] == "CPU"


def allowed(job: dict, lane: str, settings: dict) -> bool:
    """Whether this lane may run the job, ignoring whether the lane is busy."""
    if not settings[lane]["enabled"]:
        return False
    if job["platform"] in LANES:
        return job["platform"] == lane
    if lane == "colab" and is_cpu(job) and not settings["colab"]["cpu_jobs"]:
        return False
    return True


def deps_state(job: dict, state: dict) -> str:
    """'ok' when every dependency is done, 'blocked' if one failed or was cancelled, else 'waiting'."""
    recs = state.get("jobs", {})
    for dep in job["depends_on"]:
        st = recs.get(dep, {}).get("status")
        if st in ("failed", "cancelled", "blocked"):
            return "blocked"
        if st != "done":
            return "waiting"
    return "ok"


def pick(queue: list, state: dict, lane: str, settings: dict):
    """First runnable job for a free lane, in queue order, or None."""
    recs = state.get("jobs", {})
    busy = {v.get("job") for v in state.get("lanes", {}).values() if v}
    for job in queue:
        rec = recs.get(job["id"], {})
        if job.get("cancel") or rec.get("status") in FINISHED + ("blocked",) or job["id"] in busy:
            continue
        if deps_state(job, state) != "ok":
            continue
        pinned = rec.get("platform")
        if pinned and pinned != lane:
            continue
        if allowed(job, lane, settings):
            return job
    return None


MARKERS = ((vm_core.MARK_DONE, "done"), (vm_core.MARK_EXIT, "job_error"), (vm_core.MARK_SETUP_FAILED, "setup_error"))


def classify(tail: str) -> str:
    """Outcome of one attempt from the end of its log. The last marker wins. No marker means the
    attempt was cut: the VM was lost, the platform stopped the kernel or a timeout fired."""
    best, outcome = -1, "interrupted"
    for marker, name in MARKERS:
        i = tail.rfind(marker)
        if i > best:
            best, outcome = i, name
    return outcome


def apply_outcome(job: dict, rec: dict, outcome: str) -> tuple:
    """Record after an attempt, and the notification to send (or None)."""
    rec = {**rec, "attempts": rec.get("attempts", 0) + 1, "last_outcome": outcome}
    where = rec.get("platform", "?")
    if outcome == "done":
        rec["status"] = "done"
        return rec, f"{job['id']} finished on {where} after {rec['attempts']} attempt(s)"
    if outcome == "cancelled":
        rec["status"] = "cancelled"
        return rec, f"{job['id']} was cancelled on {where}"
    if outcome in ("job_error", "setup_error"):
        rec["failures"] = rec.get("failures", 0) + 1
        if rec["failures"] >= job["max_failures"]:
            rec["status"] = "failed"
            what = "the job" if outcome == "job_error" else "its setup"
            return rec, f"{job['id']} failed on {where}: {what} exited with an error {rec['failures']} time(s). Stopped"
        rec["status"] = "retrying"
        return rec, None
    rec["interruptions"] = rec.get("interruptions", 0) + 1
    if rec["interruptions"] > job["max_interruptions"]:
        rec["status"] = "failed"
        return rec, f"{job['id']} was cut {rec['interruptions']} times on {where}. Stopped"
    rec["status"] = "retrying"
    return rec, None
