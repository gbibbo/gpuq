"""The whole dispatcher against fake Colab and Kaggle CLIs, with bare git repos playing GitHub.
Jobs really run (on this machine), detached on "Colab" and in the foreground on "Kaggle"."""
import json
import sys
import time

from conftest import FAKES, git, show
from gpuq import dispatcher as D, gitops, vm


def make(world, monkeypatch, settings=None, **timing):
    tmp = world["tmp"]
    repo_dir = tmp / "ops-checkout"
    if not repo_dir.exists():
        git("clone", "-q", str(world["ops"]), str(repo_dir))
    monkeypatch.setenv("FAKE_STATE", str(tmp / "fake"))
    sent = world.setdefault("sent", [])
    monkeypatch.setattr(D.notify, "send", lambda topic, title, msg, **k: sent.append(f"{title}: {msg}") or True)
    t = {"tick_s": 0, "poll_s": 0.2, "watch_s": 0.2, "log_publish_s": 0, "code_check_s": 10 ** 6, "beat_s": 0,
         "poll_failures": 3, "exec_silence_s": 60, **timing}
    kin = tmp / "kaggle" / "input" / "gpuq-credentials"
    kin.mkdir(parents=True, exist_ok=True)
    (kin / vm.CREDENTIALS_FILE).write_text(json.dumps(world["creds"]))  # the attached private dataset
    return D.Dispatcher(settings or world["settings"], repo_dir, tmp / "state", world["secrets"],
                        colab_cmd=[sys.executable, str(FAKES / "fake_colab.py")],
                        kaggle_cmd=[sys.executable, str(FAKES / "fake_kaggle.py")], colab_root=str(tmp / "vm"),
                        kaggle_paths=vm.kaggle_paths(str(tmp / "kaggle" / "working"), str(tmp / "kaggle" / "input"),
                                                     str(tmp / "kaggle" / "scratch")),
                        timing=t, vm_extra={"home": str(tmp / "home"), "install_rclone": False,
                                            "check_internet": False})


def set_queue(world, *entries):
    b = gitops.Branch(str(world["ops"]), "jobs", world["tmp"] / "laptop-jobs", None, backoff_s=0)
    b.publish({"queue.json": json.dumps({"jobs": list(entries)})}, message="queue")


def run_until(d, cond, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d.step()
        if cond(d.state["jobs"]):
            return
        time.sleep(0.1)
    raise AssertionError("timeout. state: " + json.dumps(d.state, default=str)[:3000])


def statuses(recs, *ids):
    return tuple(recs.get(i, {}).get("status") for i in ids)


def test_jobs_run_on_both_lanes_with_dependencies(world, monkeypatch):
    set_queue(world, {"id": "a", "cmd": "python job.py", "platform": "colab"},
              {"id": "b", "cmd": "python job.py", "platform": "kaggle"},
              {"id": "c", "cmd": "python job.py", "gpu": "CPU", "depends_on": ["a", "b"]})
    d = make(world, monkeypatch)
    run_until(d, lambda r: statuses(r, "a", "b", "c") == ("done",) * 3)
    recs = d.state["jobs"]
    assert (recs["a"]["platform"], recs["b"]["platform"], recs["c"]["platform"]) == ("colab", "kaggle", "kaggle")
    assert all(recs[j]["attempts"] == 1 for j in "abc")
    status = json.loads(show(world["ops"], "runs", "_gpuq/status.json"))
    assert {j: status["jobs"][j]["status"] for j in "abc"} == {"a": "done", "b": "done", "c": "done"}
    for j, where in (("a", "colab"), ("b", "kaggle"), ("c", "kaggle")):
        text = show(world["ops"], "runs", f"{j}/attempt-01.log").decode()
        assert f"job running on {where} attempt 1" in text and "job done" in text and "[gpuq] job finished" in text
        assert "supersecretvalue42" not in text and "secret is ***" in text          # secrets never published
        assert json.loads((world["remote"] / "out" / f"{j}.json").read_text()) == {"job": j}
    assert "supersecretvalue42" not in json.dumps(status) and "ghp_faketoken" not in json.dumps(status)
    kmeta = json.loads((world["tmp"] / "fake" / "kaggle" / "book.json").read_text())["last_meta"]
    assert kmeta["enable_gpu"] is False and kmeta["is_private"] is True and kmeta["dataset_sources"] == ["me/gpuq-credentials"]
    assert sum("finished on" in m for m in world["sent"]) == 3
    assert not (world["tmp"] / "vm" / vm.CREDENTIALS_FILE).exists()            # deleted on the VM after reading


def test_cuts_are_relaunched_and_errors_stop_after_max_failures(world, monkeypatch):
    monkeypatch.setenv("FAKE_COLAB_LOSE_START", "1")     # the first Colab VM is lost before the job starts
    monkeypatch.setenv("FAKE_KAGGLE_CUT", "1")           # the first Kaggle run dies without output
    set_queue(world, {"id": "x", "cmd": "python job.py", "platform": "colab"},
              {"id": "y", "cmd": "python job.py --rc 3", "platform": "colab"},
              {"id": "w", "cmd": "python job.py", "platform": "kaggle"},
              {"id": "z", "cmd": "python job.py", "setup": "exit 5", "platform": "kaggle"})
    d = make(world, monkeypatch)
    run_until(d, lambda r: statuses(r, "x", "y", "w", "z") == ("done", "failed", "done", "failed"))
    r = d.state["jobs"]
    assert r["x"]["interruptions"] == 1 and r["x"]["attempts"] == 2
    assert r["w"]["interruptions"] == 1 and r["w"]["attempts"] == 2
    assert r["y"]["failures"] == 2 and r["y"]["last_outcome"] == "job_error"
    assert r["z"]["failures"] == 2 and r["z"]["last_outcome"] == "setup_error"
    assert any("y failed on colab: the job exited" in m for m in world["sent"])
    assert any("z failed on kaggle: its setup exited" in m for m in world["sent"])
    assert "[gpuq] job exited with code 3" in show(world["ops"], "runs", "y/attempt-02.log").decode()


def test_no_gpu_on_colab_sends_an_auto_job_to_kaggle_and_cpu_jobs_skip_colab(world, monkeypatch):
    monkeypatch.setenv("FAKE_COLAB_NO_GPU", "1")
    set_queue(world, {"id": "g", "cmd": "python job.py"})
    d = make(world, monkeypatch)
    run_until(d, lambda r: statuses(r, "g") == ("done",))
    assert d.state["jobs"]["g"]["platform"] == "kaggle" and d.state["jobs"]["g"]["attempts"] == 1
    assert d.state["unavailable"]["colab"] > time.time()     # the Colab lane rests


def test_kaggle_without_quota_rests_and_notifies(world, monkeypatch):
    monkeypatch.setenv("FAKE_KAGGLE_NO_QUOTA", "1")
    set_queue(world, {"id": "q", "cmd": "python job.py", "platform": "kaggle"})
    d = make(world, monkeypatch)
    for _ in range(3):
        d.step()
    assert d.state["jobs"]["q"]["status"] == "pending" and d.state["unavailable"]["kaggle"] > time.time()
    assert any(m.startswith("Kaggle quota") for m in world["sent"])


def test_cancel_a_running_colab_job_and_a_pending_one(world, monkeypatch):
    set_queue(world, {"id": "long", "cmd": "python job.py --sleep 4", "platform": "colab"},
              {"id": "later", "cmd": "python job.py", "platform": "colab"})
    d = make(world, monkeypatch)
    run_until(d, lambda r: (d.state["lanes"]["colab"] or {}).get("job") == "long" and
              "job running" in d.attempt_log("long", 1).read_text())
    set_queue(world, {"id": "long", "cmd": "python job.py --sleep 4", "platform": "colab", "cancel": True},
              {"id": "later", "cmd": "python job.py", "platform": "colab", "cancel": True})
    run_until(d, lambda r: statuses(r, "long", "later") == ("cancelled", "cancelled"))
    assert d.state["jobs"]["long"]["last_outcome"] == "cancelled" and "attempts" not in d.state["jobs"]["later"]


def test_a_restart_during_a_colab_attempt_counts_as_a_cut_and_the_job_is_relaunched(world, monkeypatch):
    set_queue(world, {"id": "r", "cmd": "python job.py --sleep 1", "platform": "colab"})
    d = make(world, monkeypatch)
    run_until(d, lambda r: (d.state["lanes"]["colab"] or {}).get("job") == "r")
    d.colab_thread.join(timeout=60)                      # let the old attempt end before "restarting"
    d2 = make(world, monkeypatch)                        # new process: same state file, no thread
    run_until(d2, lambda r: statuses(r, "r") == ("done",))
    assert d2.state["jobs"]["r"]["interruptions"] == 1 and d2.state["jobs"]["r"]["attempts"] == 2


def test_follow_gives_up_on_a_vm_that_stopped_answering(tmp_path):
    good = f"{vm.BEGIN}\n[t] one\n\n{vm.MARK} 8 running 9000 12000 70.5 35,1200\n"
    answers = iter([(0, good)] + [(1, "RuntimeError: Connection was lost.")] * 3)
    outcome, tail = D.follow(lambda off: next(answers), tmp_path / "a.log", sleep=lambda s: None, max_failures=3,
                             beat_s=0)
    assert outcome == "interrupted" and "[t] one" in tail and "the VM does not answer" in tail
    assert "RAM free 9000 of 12000 MB, disk free 70.5 GB, GPU util%,MiB 35,1200" in tail
    offsets = []
    answers = iter([(0, good), (1, "timeout"), (0, f"{vm.BEGIN}\n[t] two\n\n{vm.MARK} 16 exit 0 9000 12000 70.5 -\n")])
    outcome, tail = D.follow(lambda off: offsets.append(off) or next(answers), tmp_path / "b.log",
                             sleep=lambda s: None)
    assert outcome == "done" and offsets == [0, 8, 8] and "[t] two" in tail
    answers = iter([(0, good), (1, "[colab] Session 'gpuq' appears to be lost (404/401). Cleaning up.")])
    outcome, tail = D.follow(lambda off: next(answers), tmp_path / "c.log", sleep=lambda s: None, max_failures=6)
    assert outcome == "interrupted" and "poll failed 1/6" in tail      # a released VM is not polled five more times


def test_watchdog_cuts_a_silent_exec_and_spares_a_chatty_one(tmp_path):
    silent = [sys.executable, "-c", "print('hi', flush=True); import time; time.sleep(30)"]
    rc, tail = D.run_watched(silent, 60, tmp_path / "silent.log", silence_s=1.5, watch_s=0.2)
    assert rc != 0 and "hi" in tail and "no output for" in tail
    chatty = [sys.executable, "-c", "import time\nfor i in range(6):\n print(i, flush=True); time.sleep(0.5)\n"]
    rc, tail = D.run_watched(chatty, 60, tmp_path / "chatty.log", silence_s=1.5, watch_s=0.2)
    assert rc == 0 and "stopped the command" not in tail
