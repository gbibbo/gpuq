"""The dispatcher. Runs the jobs in your queue on free Colab and Kaggle machines.

It runs inside the gpuq container on your always-on machine. docker/bootstrap.py starts it from a
checkout of your ops repo. Two lanes work in parallel, one job at a time each.

    colab   through the official Colab CLI (new, upload, exec, stop), in a thread. A short exec
            starts the job detached on the VM, and a poll exec every minute brings back the new part
            of its log (see vm.py).
    kaggle  through the Kaggle CLI. It pushes a private notebook, follows its status and downloads
            its log at the end.

Every minute it reads the queue (branch jobs, queue.json), starts what can start, follows what is
running and relaunches interrupted attempts. Your job has to resume by itself (docs/WRITING_JOBS.md).
It publishes its state to runs/_gpuq/status.json and each attempt's log to runs/<job>/attempt-NN.log,
and sends a notification when a job ends, fails or needs you. When the ops repo's branch moves and
the Colab lane is empty it exits, and Docker restarts it on the new code.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import gitops, jobs, kaggle, notify, secrets, settings as settings_mod, vm, vm_core

TIMING = {
    "tick_s": 60, "poll_s": 60, "log_publish_s": 600, "code_check_s": 300,
    "start_timeout_s": 3600,        # setup and start of the detached job (clone, pip)
    "exec_silence_s": 1200,         # a start exec this long without output has lost its VM
    "watch_s": 30, "poll_timeout_s": 240,
    "poll_failures": 6,             # consecutive failed polls before the attempt counts as cut
    "beat_s": 300,
    "kaggle_max_run_s": 13 * 3600,  # Kaggle stops sessions at 12 h
    "colab_no_gpu_cooldown_s": 1800, "kaggle_no_quota_cooldown_s": 6 * 3600, "auth_cooldown_s": 3600,
    "no_gpu_notify_after_s": 3 * 3600,
}
AUTH_MARKERS = ("invalid_grant", "unauthorized", "401", "reauth", "refresh token", "credentials",
                "authorization code", "to authorize colab-cli")
# The Colab CLI says so when the backend has already released the VM. No later poll can reach it.
LOST_MARKERS = ("appears to be lost", "not found")
STATUS_FILE = "_gpuq/status.json"


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_json(p: Path, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(p: Path, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)


def run(cmd: list, timeout: float, log_path=None, env=None) -> tuple:
    """Exit code and the end of the output. stdin is closed, so a CLI that asks a question fails
    instead of waiting forever."""
    if log_path is None:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                                 stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
            return res.returncode, (res.stdout + res.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            return -9, "timeout"
    with open(log_path, "a", encoding="utf-8") as fh:
        try:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, env=env,
                                stdin=subprocess.DEVNULL).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return rc, Path(log_path).read_text(encoding="utf-8", errors="replace")[-6000:]


def run_watched(cmd: list, timeout: float, log_path: Path, silence_s: float, watch_s: float = 30) -> tuple:
    """Like run with a log file, but kills the command when its log stops growing for silence_s.
    `colab exec` never returns when the connection to the VM is lost, so silence is the only sign."""
    log_path = Path(log_path)
    with open(log_path, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        start = last = time.time()
        size = log_path.stat().st_size
        while proc.poll() is None:
            time.sleep(watch_s)
            now = time.time()
            if log_path.stat().st_size != size:
                size, last = log_path.stat().st_size, now
            reason = ("timeout" if now - start > timeout else
                      f"no output for {int(now - last)} s" if now - last > silence_s else None)
            if reason:
                proc.kill()
                proc.wait()
                fh.write(f"\n[dispatcher] stopped the command: {reason}\n")
                fh.flush()
                break
        rc = proc.returncode if proc.returncode is not None else -9
    return rc, log_path.read_text(encoding="utf-8", errors="replace")[-6000:]


def run_capture(cmd: list, timeout: float, env=None) -> tuple:
    """Exit code and the whole output."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=timeout, env=env, stdin=subprocess.DEVNULL)
        return res.returncode, res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        return -9, "timeout"


def follow(poll, log_path: Path, sleep=time.sleep, now=time.time, poll_s: float = 60, max_failures: int = 6,
           deadline_s: float = 12 * 3600, beat_s: float = 300, cancelled=lambda: False) -> tuple:
    """Follow a job detached on a Colab VM. poll(offset) -> (rc, output) runs one poll exec. The new log
    text goes to log_path, with a heartbeat line every beat_s. Returns (outcome, tail)."""
    log_path = Path(log_path)
    offset, failures, start, last_beat = 0, 0, now(), None
    with open(log_path, "a", encoding="utf-8") as fh:
        while True:
            sleep(poll_s)
            if cancelled():
                fh.write("[dispatcher] the job was cancelled\n")
                fh.close()
                return "cancelled", log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            rc, out = poll(offset)
            p = vm.parse_poll(out) if rc == 0 else None
            if p is None:
                failures += 1
                lost = any(m in out.lower() for m in LOST_MARKERS)
                fh.write(f"[dispatcher {time.strftime('%H:%M:%S', time.gmtime())}] poll failed "
                         f"{failures}/{max_failures}: {out.strip()[-300:]}\n")
                fh.flush()
                if lost or failures >= max_failures:
                    fh.write("[dispatcher] the VM does not answer, the attempt counts as cut\n")
                    break
                continue
            failures = 0
            fh.write(p["text"])
            offset = p["offset"]
            if last_beat is None or now() - last_beat >= beat_s:
                last_beat = now()
                fh.write(f"[heartbeat {time.strftime('%H:%M:%S', time.gmtime())}] RAM free {p['mem_avail']} of "
                         f"{p['mem_total']} MB, disk free {p['disk_gb']:.1f} GB, GPU util%,MiB {p['gpu']}\n")
            if p["rc"] is not None:
                fh.write(f"{vm_core.MARK_DONE}\n" if p["rc"] == 0 else f"{vm_core.MARK_EXIT} {p['rc']}\n")
                fh.close()
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
                return jobs.classify(tail), tail
            fh.flush()
            if now() - start > deadline_s:
                fh.write("[dispatcher] the job passed its timeout\n")
                break
    return "interrupted", log_path.read_text(encoding="utf-8", errors="replace")[-6000:]


class Dispatcher:
    def __init__(self, settings: dict, repo_dir, state_dir, secrets_dir=None, colab_cmd=("colab",),
                 kaggle_cmd=("kaggle",), colab_root: str = vm.COLAB_ROOT, kaggle_paths: dict = None,
                 timing: dict = None, vm_extra: dict = None):
        self.s = settings
        self.t = {**TIMING, **{k: settings["dispatcher"][k] for k in ("tick_s", "poll_s", "log_publish_s")},
                  **(timing or {})}
        self.repo_dir, self.state_dir = Path(repo_dir), Path(state_dir)
        self.secrets_dir = secrets.folder(secrets_dir)
        self.creds_path = self.secrets_dir / secrets.CREDENTIALS
        self.creds = secrets.load_credentials(self.secrets_dir)
        self.token = self.creds.get("GITHUB_TOKEN")
        ktok = secrets.read(secrets.KAGGLE_TOKEN, self.secrets_dir)
        self.topic = secrets.read(secrets.NTFY_TOPIC, self.secrets_dir)
        self.frags = secrets.fragments(self.creds, ktok or "", self.topic or "")
        self.kaggle_env = {**os.environ, "KAGGLE_API_TOKEN": ktok} if ktok else None
        self.colab_cmd, self.kaggle_cmd = list(colab_cmd), list(kaggle_cmd)
        self.colab_root = colab_root
        self.kaggle_paths = kaggle_paths or vm.kaggle_paths()
        self.vm_extra = vm_extra or {}
        ops = settings["github"]["ops_repo"]
        self.ops_url = gitops.repo_url(ops)
        clones = self.state_dir / "clones"
        self.queue_branch = gitops.Branch(ops, "jobs", clones / "jobs", self.token, redact=self.frags)
        self.runs = gitops.Branch(ops, "runs", clones / "runs", self.token, redact=self.frags)
        self.state_path = self.state_dir / "state.json"
        self.state = load_json(self.state_path, {})
        for key, default in (("jobs", {}), ("lanes", {lane: None for lane in jobs.LANES}), ("unavailable", {})):
            self.state.setdefault(key, default)
        self.lock = threading.Lock()
        self.colab_thread, self.colab_result = None, None
        self.colab_cancel = threading.Event()
        sha = self._git("rev-parse", "HEAD").strip()
        self.sha = sha if re.fullmatch(r"[0-9a-f]{40}", sha) else "unknown"
        self.next_code_check = time.time() + self.t["code_check_s"]
        self.last_log_publish = {}
        self.settings_for_lanes = {**settings, "kaggle": {**settings["kaggle"],
                                                          "enabled": self.kaggle_ready()}}

    # ---------------------------------------------------------------- helpers
    def kaggle_ready(self) -> bool:
        return bool(self.s["kaggle"]["enabled"] and self.kaggle_env and self.s["kaggle"]["username"])

    def _git(self, *args) -> str:
        env = {**os.environ, **gitops.auth_env(self.token, self.ops_url)}
        try:
            return subprocess.run(["git", "-C", str(self.repo_dir), *args], capture_output=True, text=True,
                                  env=env, timeout=120, stdin=subprocess.DEVNULL).stdout
        except Exception:
            return ""

    def notify(self, title: str, msg: str):
        log(f"notify: {title}: {msg}")
        notify.send(self.topic, title, msg)

    def read_queue(self) -> list:
        raw = self.queue_branch.read("queue.json")
        return jobs.normalize(json.loads(raw.decode("utf-8")) if raw else {}, self.s)

    def publish(self, note: str):
        with self.lock:
            self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.state["dispatcher_sha"] = self.sha
            self.state["lanes_enabled"] = {"colab": bool(self.s["colab"]["enabled"]), "kaggle": self.kaggle_ready()}
            self.state["note"] = note
            save_json(self.state_path, self.state)
            text = secrets.redact(json.dumps(self.state, indent=2, default=str), self.frags)
        try:
            self.runs.publish({STATUS_FILE: text}, message=f"gpuq: {note}")
        except Exception as exc:
            log(f"publish failed: {exc!r}")

    def publish_log(self, job_id: str, attempt: int, path: Path, force: bool = False):
        key = (job_id, attempt)
        if not force and time.time() - self.last_log_publish.get(key, 0) < self.t["log_publish_s"]:
            return
        self.last_log_publish[key] = time.time()
        if not Path(path).exists():
            return
        data = Path(path).read_bytes()[-self.s["dispatcher"]["log_tail_kb"] * 1024:]
        text = secrets.redact(data.decode("utf-8", errors="replace"), self.frags)
        try:
            self.runs.publish({f"{job_id}/attempt-{attempt:02d}.log": text}, message=f"gpuq: log of {job_id}")
        except Exception as exc:
            log(f"log publish failed: {exc!r}")

    def available(self, lane: str) -> bool:
        if lane == "kaggle" and not self.kaggle_ready():
            return False
        if lane == "colab" and not self.s["colab"]["enabled"]:
            return False
        return time.time() >= self.state["unavailable"].get(lane, 0)

    def mark_unavailable(self, lane: str, seconds: float):
        self.state["unavailable"][lane] = time.time() + seconds

    def job_dir(self, job_id: str) -> Path:
        d = self.state_dir / "jobs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def attempt_log(self, job_id: str, n: int) -> Path:
        return self.job_dir(job_id) / f"attempt-{n:02d}.log"

    def params(self, job: dict, n: int, platform: str) -> dict:
        paths = vm.colab_paths(self.colab_root) if platform == "colab" else self.kaggle_paths
        ops_ref = self.sha if self.sha != "unknown" else self.s["github"]["branch"]
        return vm.params(job, n, platform, self.s["github"]["ops_repo"], ops_ref, self.s["storage"]["remote"],
                         paths, **self.vm_extra)

    def start(self, lane: str, job: dict):
        rec = self.state["jobs"].get(job["id"], {"status": "pending"})
        rec.update(status="running", platform=lane)
        rec.pop("cancel_notified", None)
        self.state["jobs"][job["id"]] = rec
        n = rec.get("attempts", 0) + 1
        self.state["lanes"][lane] = {"job": job["id"], "since": time.time(), "attempt": n}
        log(f"{lane}: starting {job['id']} attempt {n}")
        self.attempt_log(job["id"], n).write_text(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {lane} attempt {n} of "
            f"{json.dumps({k: job[k] for k in ('id', 'repo', 'ref', 'cmd', 'gpu')})}\n", encoding="utf-8")
        if lane == "colab":
            self.colab_result = None
            self.colab_cancel.clear()
            self.colab_thread = threading.Thread(target=self._colab_worker, args=(job, n), daemon=True)
            self.colab_thread.start()
        else:
            self._kaggle_push(job, n)
        self.publish(f"{lane}: running {job['id']} attempt {n}")

    def finish(self, lane: str, job: dict, outcome: str, tail: str):
        lane_rec = self.state["lanes"][lane] or {}
        self.state["lanes"][lane] = None
        n = lane_rec.get("attempt", 0)
        rec = self.state["jobs"].get(job["id"], {})
        log(f"{lane}: {job['id']} -> {outcome}")
        if n:
            self.publish_log(job["id"], n, self.attempt_log(job["id"], n), force=True)
        if outcome in ("no_gpu", "no_quota", "auth"):
            # not the job's fault: back to the queue, and the lane rests
            rec["status"] = "retrying" if rec.get("attempts") else "pending"
            if not rec.get("attempts"):
                rec.pop("platform", None)  # an auto job that never ran may go to the other lane
            if outcome == "no_gpu":
                self.mark_unavailable(lane, self.t["colab_no_gpu_cooldown_s"])
                since = self.state.setdefault("no_gpu_since", time.time())
                if time.time() - since > self.t["no_gpu_notify_after_s"] and not self.state.get("no_gpu_notified"):
                    self.notify("Colab gives no GPU", "Colab has not given a GPU for more than 3 h. Still trying")
                    self.state["no_gpu_notified"] = True
            elif outcome == "no_quota":
                self.mark_unavailable(lane, self.t["kaggle_no_quota_cooldown_s"])
                self.notify("Kaggle quota", "Kaggle takes no more GPU runs for now. Trying again in 6 h")
            else:
                self.mark_unavailable(lane, self.t["auth_cooldown_s"])
                self.notify("Colab login needed", "The Colab CLI lost its Google login. On the dispatcher machine "
                            "run:\ndocker exec -it gpuq colab sessions")
            self.state["jobs"][job["id"]] = rec
            self.publish(f"{lane}: {job['id']} {outcome}")
            return
        if lane == "colab":
            self.state.pop("no_gpu_since", None)
            self.state.pop("no_gpu_notified", None)
        rec, message = jobs.apply_outcome(job, rec, outcome)
        rec["last_tail"] = secrets.redact(tail[-1500:], self.frags)
        self.state["jobs"][job["id"]] = rec
        self.publish(f"{lane}: {job['id']} {outcome}")
        if message:
            self.notify("gpuq", message)

    # ---------------------------------------------------------------- colab lane
    def _colab(self, *args, timeout=600, log_path=None):
        return run([*self.colab_cmd, *args], timeout, log_path)

    def _colab_worker(self, job: dict, n: int):
        try:
            self.colab_result = self._colab_attempt(job, n)
        except Exception as exc:
            self.colab_result = ("interrupted", f"dispatcher exception: {exc!r}")

    def _colab_attempt(self, job: dict, n: int) -> tuple:
        session = self.s["colab"]["session"]
        log_path = self.attempt_log(job["id"], n)
        gpu = [] if jobs.is_cpu(job) else ["--gpu", job["gpu"]]
        rc, out = self._colab("new", "-s", session, *gpu, timeout=900, log_path=log_path)
        if rc != 0:
            self._colab("stop", "-s", session, timeout=180)
            return ("auth" if any(m in out.lower() for m in AUTH_MARKERS) else "no_gpu"), out
        p = self.params(job, n, "colab")
        rc, out = self._colab("upload", "-s", session, str(self.creds_path), p["creds_path"], timeout=300,
                              log_path=log_path)
        if rc != 0:
            self._colab("stop", "-s", session, timeout=180)
            return "interrupted", out
        script = self.job_dir(job["id"]) / "start.py"
        script.write_text(vm.start_script(p), encoding="utf-8")
        rc, out = run_watched([*self.colab_cmd, "exec", "-s", session, "--timeout", str(self.t["start_timeout_s"]),
                               "-f", str(script)], self.t["start_timeout_s"] + 300, log_path,
                              self.t["exec_silence_s"], self.t["watch_s"])
        if vm_core.MARK_STARTED not in out:  # setup failed or the VM was lost before the job started
            self._colab("stop", "-s", session, timeout=180)
            return jobs.classify(out), out
        poll_file = self.job_dir(job["id"]) / "poll.py"
        poll_file.write_text(vm.poll_script(p["log"], p["rc"]), encoding="utf-8")

        def poll(offset):
            return run_capture([*self.colab_cmd, "exec", "-s", session, "--timeout", "120", "--env",
                                f"GPUQ_OFFSET={offset}", "-f", str(poll_file)], self.t["poll_timeout_s"])

        outcome, tail = follow(poll, log_path, poll_s=self.t["poll_s"], max_failures=self.t["poll_failures"],
                               deadline_s=float(job["timeout_h"]) * 3600, beat_s=self.t["beat_s"],
                               cancelled=self.colab_cancel.is_set)
        self._colab("stop", "-s", session, timeout=180)
        return outcome, tail

    def poll_colab(self, by_id: dict):
        lane = self.state["lanes"].get("colab")
        if not lane:
            return
        job = by_id.get(lane["job"]) or jobs.normalize({"jobs": [{"id": lane["job"], "cmd": "?"}]}, self.s)[0]
        if self.colab_thread is None:  # dispatcher restarted mid-attempt: the attempt is lost
            self._colab("stop", "-s", self.s["colab"]["session"], timeout=180)
            self.finish("colab", job, "interrupted", "the dispatcher restarted during the attempt")
            return
        if job.get("cancel"):
            self.colab_cancel.set()
        if not self.colab_thread.is_alive():
            outcome, tail = self.colab_result or ("interrupted", "no result")
            self.colab_thread = None
            self.finish("colab", job, outcome, tail)
            return
        self.publish_log(job["id"], lane["attempt"], self.attempt_log(job["id"], lane["attempt"]))

    # ---------------------------------------------------------------- kaggle lane
    def _kaggle(self, *args, timeout=300):
        return run([*self.kaggle_cmd, *args], timeout, env=self.kaggle_env)

    def kernel_ref(self) -> str:
        return f"{self.s['kaggle']['username']}/{self.s['kaggle']['kernel']}"

    def _kaggle_push(self, job: dict, n: int):
        folder = self.job_dir(job["id"]) / f"kaggle-kernel-{n:02d}"
        k = self.s["kaggle"]
        kaggle.build_kernel(folder, vm.start_script(self.params(job, n, "kaggle")), k["username"], k["kernel"],
                            k["dataset"], job["gpu"])
        rc, out = self._kaggle("kernels", "push", "-p", str(folder))
        lane = self.state["lanes"]["kaggle"]
        with open(self.attempt_log(job["id"], n), "a", encoding="utf-8") as fh:
            fh.write(f"[dispatcher] kaggle kernels push: {out.strip()[-600:]}\n")
        if rc != 0 or "successfully pushed" not in out.lower():
            lane["push_error"] = out[-1500:]
        else:
            m = re.search(r"version (\d+)", out)
            lane["version"] = int(m.group(1)) if m else None

    def poll_kaggle(self, by_id: dict):
        lane = self.state["lanes"].get("kaggle")
        if not lane:
            return
        job = by_id.get(lane["job"]) or jobs.normalize({"jobs": [{"id": lane["job"], "cmd": "?"}]}, self.s)[0]
        n, log_path = lane.get("attempt", 0), self.attempt_log(lane["job"], lane.get("attempt", 0))
        if lane.get("push_error"):
            err = lane["push_error"]
            self.finish("kaggle", job, "no_quota" if any(m in err.lower() for m in kaggle.QUOTA_MARKERS)
                        else "interrupted", err)
            return
        if job.get("cancel") and not self.state["jobs"].get(job["id"], {}).get("cancel_notified"):
            self.state["jobs"].setdefault(job["id"], {})["cancel_notified"] = True
            self.notify("gpuq", f"{job['id']} is running on Kaggle, which cannot be stopped from the CLI. Stop it at "
                        f"https://www.kaggle.com/code/{self.kernel_ref()} (Cancel run). gpuq will record it as cancelled")
        rc, out = self._kaggle("kernels", "status", self.kernel_ref(), timeout=120)
        st = kaggle.state(out) if rc == 0 else "running"
        if st == "running":
            if time.time() - lane["since"] > self.t["kaggle_max_run_s"]:
                self.finish("kaggle", job, "interrupted", "the Kaggle run passed 13 h")
            elif time.time() - self.last_log_publish.get((job["id"], n), 0) >= self.t["log_publish_s"]:
                live_path = self.job_dir(job["id"]) / f"kaggle-live-{n:02d}.log"
                rc_l, live = self._kaggle("kernels", "logs", self.kernel_ref(), timeout=120)
                if rc_l == 0 and live.strip():  # best effort: the log so far, when Kaggle gives it
                    live_path.write_text(log_path.read_text(encoding="utf-8", errors="replace") +
                                         kaggle.log_text(live.encode("utf-8")), encoding="utf-8")
                self.publish_log(job["id"], n, live_path if live_path.exists() else log_path)
            return
        dest = self.job_dir(job["id"]) / f"kaggle-output-{n:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        self._kaggle("kernels", "output", self.kernel_ref(), "-p", str(dest), "-o", "-q", timeout=900)
        logs = sorted(dest.glob("*.log"))
        text = kaggle.log_text(logs[0].read_bytes()) if logs else out
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(text)
        tail = text[-6000:]
        if job.get("cancel") or st == "cancelled":
            self.finish("kaggle", job, "cancelled", tail)
        elif st != "complete" and any(m in tail.lower() for m in kaggle.QUOTA_MARKERS):
            self.finish("kaggle", job, "no_quota", tail)
        else:
            self.finish("kaggle", job, jobs.classify(tail), tail)

    # ---------------------------------------------------------------- main loop
    def code_moved(self) -> bool:
        """True when the ops repo's branch moved and the Colab lane is empty. A Kaggle run keeps going
        and the new dispatcher resumes following it."""
        if time.time() < self.next_code_check or self.state["lanes"].get("colab") or self.sha == "unknown":
            return False
        self.next_code_check = time.time() + self.t["code_check_s"]
        env = {**os.environ, **gitops.auth_env(self.token, self.ops_url)}
        rc, out = run(["git", "ls-remote", self.ops_url, f"refs/heads/{self.s['github']['branch']}"], 120, env=env)
        head = out.split()[0] if rc == 0 and out.strip() else ""
        if not head or head == self.sha:
            return False
        log("the ops repo moved, restarting on the new code")
        self.publish("restarting on new code")
        return True

    def housekeeping(self, queue: list):
        """Cancellations of jobs that are not running, and jobs blocked by a failed dependency."""
        for j in queue:
            rec = self.state["jobs"].get(j["id"])
            busy = any(v and v.get("job") == j["id"] for v in self.state["lanes"].values())
            status = (rec or {}).get("status")
            if j.get("cancel") and not busy and status not in jobs.FINISHED:
                self.state["jobs"][j["id"]] = {**(rec or {}), "status": "cancelled"}
                self.publish(f"cancelled {j['id']}")
            elif not busy and status not in jobs.FINISHED + ("blocked",) and jobs.deps_state(j, self.state) == "blocked":
                self.state["jobs"][j["id"]] = {**(rec or {}), "status": "blocked"}
                self.publish(f"blocked {j['id']}")
                self.notify("gpuq", f"{j['id']} is blocked because one of its dependencies failed or was cancelled")

    def step(self) -> bool:
        """One tick. Returns False when the dispatcher should exit to restart on new code."""
        try:
            queue = self.read_queue()
        except Exception as exc:
            log(f"cannot read the queue: {exc!r}")
            return True
        by_id = {j["id"]: j for j in queue}
        self.housekeeping(queue)
        self.poll_colab(by_id)
        self.poll_kaggle(by_id)
        if self.code_moved():  # after the polls, so a finished attempt is recorded first
            return False
        lane_settings = {**self.s, "kaggle": {**self.s["kaggle"], "enabled": self.kaggle_ready()}}
        for lane in jobs.LANES:
            if self.state["lanes"].get(lane) or not self.available(lane):
                continue
            job = jobs.pick(queue, self.state, lane, lane_settings)
            if job is not None:
                self.start(lane, job)
        if not any(self.state["lanes"].values()) and self.state.get("note") != "idle":
            self.publish("idle")
        return True

    def run(self, max_ticks=None) -> int:
        log(f"dispatcher {self.sha[:12]} starting, colab {'on' if self.s['colab']['enabled'] else 'off'}, "
            f"kaggle {'on' if self.kaggle_ready() else 'off'}")
        if self.s["colab"]["enabled"] and not self.state["lanes"].get("colab"):
            self._colab("stop", "-s", self.s["colab"]["session"], timeout=180)  # a session left by a crash
        self.publish("started")
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            if not self.step():
                return 0
            ticks += 1
            time.sleep(self.t["tick_s"])
        return 0


def main(argv=None) -> int:
    repo_dir = Path(os.environ.get("GPUQ_REPO_DIR") or Path(__file__).resolve().parents[1])
    s = settings_mod.load(repo_dir / settings_mod.FILE)
    problems = settings_mod.check(s)
    if problems:
        for p in problems:
            log(f"settings: {p}")
        return 2
    state_dir = Path(os.environ.get("GPUQ_STATE_DIR") or "/data/state")
    d = Dispatcher(s, repo_dir, state_dir)
    if not d.token:
        log("no GITHUB_TOKEN in the credentials. Run gpuq setup")
        return 2
    return d.run()


if __name__ == "__main__":
    sys.exit(main())
