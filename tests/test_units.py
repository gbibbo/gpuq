"""Queue logic, settings, secrets, storage, branches and the VM code, without a dispatcher."""
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import ROOT, bare_repo, git, make_settings, show
from gpuq import gitops, jobs, secrets, settings as settings_mod, storage, vm, vm_core


@pytest.fixture
def s(tmp_path):
    return make_settings(tmp_path / "ops.git", tmp_path / "code.git", tmp_path / "remote")


# ------------------------------------------------------------------ queue logic

def test_queue_defaults_and_invalid_entries(s):
    q = jobs.normalize({"jobs": [{"id": "a", "cmd": "python a.py"}, {"id": "a", "cmd": "dup"}, {"cmd": "no id"},
                                 {"id": "bad id!", "cmd": "x"}, {"id": "nocmd"}, {"id": "b", "cmd": "x", "gpu": "cpu",
                                 "platform": "nowhere", "depends_on": None, "setup": ""}]}, s)
    assert [j["id"] for j in q] == ["a", "b"]
    a, b = q
    assert a["repo"] == s["github"]["code_repo"] and a["ref"] == "main" and a["gpu"] == "T4" and a["setup"] is None
    assert a["max_failures"] == 2 and a["max_interruptions"] == 10 and a["timeout_h"] == 12
    assert b["gpu"] == "CPU" and b["platform"] == "auto" and b["depends_on"] == [] and b["setup"] == ""
    assert jobs.normalize(None, s) == []


def test_lanes_pinning_dependencies_and_cpu_routing(s):
    q = jobs.normalize({"jobs": [{"id": "base", "cmd": "x", "platform": "colab"},
                                 {"id": "lora", "cmd": "x", "platform": "kaggle"},
                                 {"id": "joint", "cmd": "x", "depends_on": ["base", "lora"]},
                                 {"id": "cpu", "cmd": "x", "gpu": "CPU"}, {"id": "extra", "cmd": "x"}]}, s)
    st = {"jobs": {}, "lanes": {"colab": None, "kaggle": None}}
    assert jobs.pick(q, st, "colab", s)["id"] == "base" and jobs.pick(q, st, "kaggle", s)["id"] == "lora"
    st["lanes"] = {"colab": {"job": "base"}, "kaggle": {"job": "lora"}}
    assert jobs.pick(q, st, "colab", s)["id"] == "extra"          # joint waits, cpu jobs skip Colab
    assert jobs.pick(q, st, "kaggle", s)["id"] == "cpu"
    s2 = {**s, "colab": {**s["colab"], "cpu_jobs": True}}
    assert jobs.pick(q, st, "colab", s2)["id"] == "cpu"
    st = {"jobs": {"base": {"status": "done"}, "lora": {"status": "retrying", "platform": "kaggle"}}, "lanes": {}}
    assert jobs.deps_state(q[2], st) == "waiting" and jobs.pick(q, st, "colab", s)["id"] == "extra"
    st["jobs"]["lora"]["status"] = "done"
    assert jobs.pick(q, st, "colab", s)["id"] == "joint"
    st["jobs"]["lora"]["status"] = "failed"
    assert jobs.deps_state(q[2], st) == "blocked"
    s3 = {**s, "kaggle": {**s["kaggle"], "enabled": False}}
    assert not jobs.allowed(q[1], "kaggle", s3) and not jobs.allowed(q[1], "colab", s3)


def test_classify_by_the_last_marker():
    assert jobs.classify("...\n[gpuq] job finished\n") == "done"
    assert jobs.classify("[gpuq] job finished\n...\n[gpuq] job exited with code 3\n") == "job_error"
    assert jobs.classify("[gpuq] setup failed: git fetch failed\n") == "setup_error"
    assert jobs.classify("Connection lost") == "interrupted" and jobs.classify("") == "interrupted"


def test_retry_policy(s):
    job = jobs.normalize({"jobs": [{"id": "p", "cmd": "x", "max_failures": 2, "max_interruptions": 2}]}, s)[0]
    rec, msg = jobs.apply_outcome(job, {"platform": "colab"}, "interrupted")
    assert rec["status"] == "retrying" and msg is None
    rec, _ = jobs.apply_outcome(job, rec, "interrupted")
    rec, msg = jobs.apply_outcome(job, rec, "interrupted")
    assert rec["status"] == "failed" and "cut 3 times on colab" in msg
    rec, msg = jobs.apply_outcome(job, {}, "job_error")
    assert rec["status"] == "retrying" and msg is None
    rec, msg = jobs.apply_outcome(job, rec, "setup_error")
    assert rec["status"] == "failed" and "its setup exited" in msg
    rec, msg = jobs.apply_outcome(job, {"attempts": 3, "platform": "kaggle"}, "done")
    assert rec["status"] == "done" and "kaggle after 4 attempt" in msg


# ------------------------------------------------------------------ settings and secrets

def test_settings_file_env_override_and_checks(tmp_path):
    p = tmp_path / "gpuq.toml"
    p.write_text('[github]\nops_repo = "me/gpuq"\n[kaggle]\nusername = ""\n[storage]\nremote = "gdrive:gpuq"\n')
    s = settings_mod.load(p, env={})
    assert s["github"]["code_repo"] == "me/gpuq" and s["jobs"]["gpu"] == "T4" and s["storage"]["remote"] == "gdrive:gpuq"
    assert settings_mod.check(s) == ["kaggle.enabled is true but kaggle.username is empty in gpuq.toml"]
    assert settings_mod.load(p, env={"GPUQ_OPS_REPO": "other/repo"})["github"]["ops_repo"] == "other/repo"
    assert "ops_repo is empty" in settings_mod.check(settings_mod.load(tmp_path / "none.toml", env={}))[0]
    example = settings_mod.load(ROOT / "gpuq.toml.example", env={})
    assert settings_mod.check(example) == [] and example["kaggle"]["username"] == "your-kaggle-username"


def test_secrets_are_private_and_redacted(tmp_path):
    conf = '[gdrive]\ntype = drive\nclient_id = 123.apps\nclient_secret = GOCSPX-verysecretvalue\n' \
           'token = {"access_token":"ya29.accesstokenvalue","refresh_token":"1//refreshtokenvalue","expiry":"x"}\n'
    creds = {"GITHUB_TOKEN": "github_pat_abcdefgh12345", "RCLONE_CONF": base64.b64encode(conf.encode()).decode(),
             "env": {"WANDB_API_KEY": "wandbkey123456"}}
    secrets.save_credentials(creds, tmp_path)
    assert secrets.load_credentials(tmp_path) == creds
    if os.name != "nt":
        assert (tmp_path / "credentials.json").stat().st_mode & 0o777 == 0o600
    frags = secrets.fragments(creds)
    text = f"x {creds['GITHUB_TOKEN']} y GOCSPX-verysecretvalue z ya29.accesstokenvalue 1//refreshtokenvalue wandbkey123456"
    out = secrets.redact(text, frags)
    assert all(v not in out for v in ("github_pat", "GOCSPX", "ya29", "refreshtoken", "wandbkey")), out
    secrets.write("ntfy_topic", "gpuq-abc", tmp_path)
    assert secrets.read("ntfy_topic", tmp_path) == "gpuq-abc" and secrets.read("missing", tmp_path) is None


# ------------------------------------------------------------------ storage

def test_put_get_and_checkpoints_keep_the_last_ones(tmp_path):
    r = storage.LocalRemote(tmp_path / "remote")
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 1000)
    info = r.put(f, "dir/a.bin")
    assert info["size"] == 1000 and r.exists("dir/a.bin") and not r.exists("dir/a.bin.part")
    assert r.get(f"dir/a.bin", tmp_path / "back.bin").read_bytes() == b"x" * 1000
    r.put_json({"k": 1}, "dir/k.json")
    assert r.get_json("dir/k.json") == {"k": 1} and r.get_json("dir/none.json") is None
    ck = storage.Checkpoints(r, "ckpt/run", keep=2)
    assert ck.latest() is None and ck.load(tmp_path / "c.pt") is None
    for step in (100, 200, 300):
        f.write_text(str(step))
        ck.save(step, f)
    assert ck.steps() == [200, 300]
    assert ck.load(tmp_path / "c.pt") == 300 and (tmp_path / "c.pt").read_text() == "300"
    assert storage.Remote.from_env({"GPUQ_REMOTE": str(tmp_path / "remote")}).exists("dir/a.bin")
    assert isinstance(storage.Remote.from_env({"GPUQ_REMOTE": "gdrive:gpuq"}), storage.RcloneRemote)
    with pytest.raises(storage.StorageError):
        storage.Remote.from_env({})


# ------------------------------------------------------------------ branches

def test_branch_creates_reads_and_survives_a_concurrent_push(tmp_path):
    bare = bare_repo(tmp_path / "ops.git", {"README.md": "ops\n"})
    a = gitops.Branch(str(bare), "runs", tmp_path / "a", None, backoff_s=0)
    b = gitops.Branch(str(bare), "runs", tmp_path / "b", None, backoff_s=0)
    assert a.read("x.txt") is None and a.head() is None                      # branch does not exist yet
    assert a.publish({"one/x.txt": "1"}, message="first")
    assert b.read("one/x.txt") == b"1" and show(bare, "runs", "README.md").startswith(b"gpuq branch")
    assert show(bare, "main", "README.md") == b"ops\n"                          # main untouched
    sync, raced = a._sync, []

    def racing_sync():                                                         # b pushes between a's sync and push
        ok = sync()
        if not raced:
            raced.append(1)
            b.publish({"two/y.txt": "2"}, message="race")
        return ok

    a._sync = racing_sync
    assert a.publish({"one/x.txt": "1b"}, message="second")
    assert show(bare, "runs", "one/x.txt") == b"1b" and show(bare, "runs", "two/y.txt") == b"2"
    assert not a.publish({"one/x.txt": "1b"})                                  # nothing new
    assert git("--git-dir", str(bare), "log", "-1", "--format=%an", "runs").strip() == "gpuq-bot"


def test_token_travels_in_a_header_only():
    env = gitops.auth_env("ghp_secret", "https://github.com/me/gpuq.git")
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert base64.b64decode(env["GIT_CONFIG_VALUE_0"].split()[-1]).decode() == "x-access-token:ghp_secret"
    assert gitops.auth_env("ghp_secret", "/local/path") == {"GIT_TERMINAL_PROMPT": "0"}
    assert gitops.repo_url("me/gpuq") == "https://github.com/me/gpuq.git"


# ------------------------------------------------------------------ the VM code

def _vm_world(tmp_path, job_py, extra_files=None):
    code = bare_repo(tmp_path / "code.git", {"job.py": job_py, **(extra_files or {})})
    ops = bare_repo(tmp_path / "ops.git", {"gpuq": ROOT / "gpuq"})
    sha = git("--git-dir", str(ops), "rev-parse", "main").strip()
    creds = tmp_path / "creds.json"
    creds.write_text(json.dumps({"GITHUB_TOKEN": "ghp_x", "env": {"FOO": "bar"}, "RCLONE_CONF": "[r]\ntype = local\n"}))
    return code, ops, sha, creds


def _params(tmp_path, code, ops, sha, creds, mode, cmd="python job.py", setup=None):
    job = {"id": "j", "repo": str(code), "ref": "main", "cmd": cmd, "setup": setup}
    paths = {"creds_path": str(creds), "workdir": str(tmp_path / "work"), "scratch": str(tmp_path / "scratch"),
             "log": str(tmp_path / "job.log"), "rc": str(tmp_path / "job.rc")}
    p = vm.params(job, 1, "colab" if mode == "detached" else "kaggle", str(ops), sha, str(tmp_path / "remote"), paths,
                  home=str(tmp_path / "home"), install_rclone=False, check_internet=False)
    p["mode"] = mode
    return p


def _run_script(tmp_path, p):
    script = tmp_path / "start.py"
    script.write_text(vm.start_script(p), encoding="utf-8")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=tmp_path)
    return r.returncode, r.stdout + r.stderr


JOB = '''import os
from gpuq.storage import Remote
print("FOO", os.environ["FOO"], "remote", os.environ["GPUQ_REMOTE"] is not None, flush=True)
Remote.from_env().put_json({"ok": True}, "r.json")
'''


def test_foreground_run_checks_out_code_and_helpers_and_reports(tmp_path):
    code, ops, sha, creds = _vm_world(tmp_path, JOB, {"requirements.txt": ""})
    rc, out = _run_script(tmp_path, _params(tmp_path, code, ops, sha, creds, "foreground"))
    assert rc == 0 and vm_core.MARK_DONE in out and "FOO bar remote True" in out, out
    assert "credentials loaded into FOO, GITHUB_TOKEN, RCLONE_CONFIG" in out and "ghp_x" not in out
    assert "setup:" in out                                            # requirements.txt was installed
    assert json.loads((tmp_path / "remote" / "r.json").read_text()) == {"ok": True}
    assert (tmp_path / "home" / ".config" / "rclone" / "rclone.conf").read_text().startswith("[r]")
    assert creds.exists()                                             # Kaggle cannot delete its dataset


def test_foreground_errors_are_marked(tmp_path):
    code, ops, sha, creds = _vm_world(tmp_path, "import sys; print('boom'); sys.exit(3)\n")
    rc, out = _run_script(tmp_path, _params(tmp_path, code, ops, sha, creds, "foreground"))
    assert rc == 3 and jobs.classify(out) == "job_error" and "boom" in out
    rc, out = _run_script(tmp_path, _params(tmp_path, code, ops, sha, creds, "foreground", setup="exit 5"))
    assert rc == 1 and jobs.classify(out) == "setup_error" and "setup command exited with code 5" in out
    p = _params(tmp_path, code, ops, sha, creds, "foreground")
    p["ref"] = "no-such-branch"
    rc, out = _run_script(tmp_path, p)
    assert jobs.classify(out) == "setup_error" and "git fetch failed" in out


def test_detached_run_is_followed_by_the_poll_script(tmp_path):
    code, ops, sha, creds = _vm_world(tmp_path, "import time\nfor i in range(3):\n print('line', i, flush=True)\n"
                                                 " time.sleep(0.3)\nprint('ñandú', flush=True)\n")
    p = _params(tmp_path, code, ops, sha, creds, "detached")
    rc, out = _run_script(tmp_path, p)
    assert rc == 0 and vm_core.MARK_STARTED in out and not creds.exists()     # deleted after reading on Colab
    poll = vm.poll_script(p["log"], p["rc"])
    offset, text, deadline = 0, "", time.time() + 60
    while time.time() < deadline:
        r = subprocess.run([sys.executable, "-c", poll], capture_output=True, text=True, encoding="utf-8",
                           env={**os.environ, "GPUQ_OFFSET": str(offset), "PYTHONIOENCODING": "utf-8"})
        parsed = vm.parse_poll(r.stdout)
        assert parsed is not None, r.stdout + r.stderr
        text, offset = text + parsed["text"], parsed["offset"]
        if parsed["rc"] is not None:
            break
        time.sleep(0.3)
    assert parsed["rc"] == 0 and text.count("line") == 3 and "ñandú" in text and parsed["mem_total"] != 0
    assert vm.parse_poll("nothing") is None and vm.parse_poll(f"{vm.BEGIN}\n{vm.MARK} x running") is None


def test_kaggle_kernel_folder_and_notebook(tmp_path):
    from gpuq import kaggle
    folder = kaggle.build_kernel(tmp_path / "k", "print('hi')\n", "me", "gpuq-launcher", "gpuq-credentials", "T4")
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "me/gpuq-launcher" and meta["is_private"] and meta["enable_internet"] and meta["enable_gpu"]
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["dataset_sources"] == ["me/gpuq-credentials"]
    nb = json.loads((folder / meta["code_file"]).read_text())
    assert nb["cells"][0]["source"] == "print('hi')\n" and nb["nbformat"] == 4
    cpu = json.loads((kaggle.build_kernel(tmp_path / "c", "x", "me", "k", "d", "CPU") / "kernel-metadata.json").read_text())
    assert cpu["enable_gpu"] is False and cpu["machine_shape"] == ""
    assert kaggle.state('me/k has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert kaggle.state('"KernelWorkerStatus.CANCEL_ACKNOWLEDGED"') == "cancelled"
    assert kaggle.state('"KernelWorkerStatus.QUEUED"') == "running"
    raw = json.dumps([{"stream_name": "stdout", "data": "a\n"}, {"stream_name": "stderr", "data": "b\n"}]).encode()
    assert kaggle.log_text(raw) == "a\nb\n" and kaggle.log_text(b"plain") == "plain"
