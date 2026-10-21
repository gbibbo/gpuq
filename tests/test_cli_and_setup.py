"""The laptop CLI against a bare repo, and the setup wizard with scripted answers."""
import json

import pytest

from conftest import bare_repo, git, make_settings, show
from gpuq import cli, gitops, secrets, setup_wizard


@pytest.fixture
def clone(tmp_path, monkeypatch):
    ops = tmp_path / "ops.git"
    bare_repo(ops, {"gpuq.toml": "placeholder\n"})
    work = tmp_path / "laptop"
    git("clone", "-q", str(ops), str(work))
    (work / "gpuq.toml").write_text(f'[github]\nops_repo = "{ops.as_posix()}"\ncode_repo = "me/research"\n'
                                    '[kaggle]\nusername = "me"\n', encoding="utf-8")
    monkeypatch.chdir(work)
    monkeypatch.setenv("GPUQ_CACHE", str(tmp_path / "cache"))
    return ops


def test_add_list_cancel_and_their_checks(clone, capsys):
    cli.main(["add", "train-1", "--cmd", "python train.py --epochs 3", "--gpu", "t4"])
    cli.main(["add", "eval-1", "--cmd", "python eval.py", "--depends-on", "train-1", "--gpu", "CPU", "--no-setup",
              "--platform", "kaggle", "--max-interruptions", "3"])
    q = json.loads(show(clone, "jobs", "queue.json"))
    assert q["jobs"][0] == {"id": "train-1", "cmd": "python train.py --epochs 3", "platform": "auto", "gpu": "T4",
                            "depends_on": []}
    assert q["jobs"][1]["setup"] == "" and q["jobs"][1]["depends_on"] == ["train-1"] and q["jobs"][1]["max_interruptions"] == 3
    with pytest.raises(SystemExit, match="already exists"):
        cli.main(["add", "train-1", "--cmd", "x"])
    with pytest.raises(SystemExit, match="unknown dependencies"):
        cli.main(["add", "x-1", "--cmd", "x", "--depends-on", "nope"])
    with pytest.raises(SystemExit, match="job ids use"):
        cli.main(["add", "bad id", "--cmd", "x"])
    cli.main(["cancel", "eval-1"])
    assert json.loads(show(clone, "jobs", "queue.json"))["jobs"][1]["cancel"] is True
    capsys.readouterr()
    cli.main(["list"])
    assert '"train-1"' in capsys.readouterr().out


def test_status_and_logs(clone, tmp_path, capsys):
    cli.main(["add", "train-1", "--cmd", "python train.py"])
    capsys.readouterr()
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "has not reported yet" in out and "train-1" in out and "pending" in out
    status = {"updated_at": "2026-10-20T10:00:00Z", "dispatcher_sha": "a" * 40, "note": "idle",
              "lanes": {"colab": None, "kaggle": None}, "lanes_enabled": {"colab": True, "kaggle": False},
              "jobs": {"train-1": {"status": "done", "platform": "colab", "attempts": 2, "last_outcome": "done"}}}
    gitops.Branch(str(clone), "runs", tmp_path / "pi-runs", None, backoff_s=0).publish(
        {"_gpuq/status.json": json.dumps(status), "train-1/attempt-02.log": "hello from the VM\n"})
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "dispatcher aaaaaaaaaaaa" in out and "kaggle  off" in out and "colab   idle" in out
    assert "train-1" in out and "done" in out and "colab" in out
    cli.main(["logs", "train-1"])
    assert capsys.readouterr().out == "hello from the VM\n"
    with pytest.raises(SystemExit, match="no log of train-1 attempt 1"):
        cli.main(["logs", "train-1", "--attempt", "1"])


class Script(setup_wizard.Io):
    def __init__(self, answers, secrets_):
        self.answers, self.secret_answers, self.lines = list(answers), list(secrets_), []
        super().__init__(ask=lambda q="": self.answers.pop(0), ask_secret=lambda q="": self.secret_answers.pop(0),
                         out=self.lines.append)


def test_setup_saves_checked_values_and_never_echoes_them(tmp_path, monkeypatch):
    s = make_settings(tmp_path / "ops.git", tmp_path / "code.git", tmp_path / "remote")
    s["github"]["ops_repo"] = "me/gpuq"
    s["storage"]["remote"] = ""
    d = tmp_path / "secrets"
    checked = []
    monkeypatch.setattr(setup_wizard, "check_github",
                        lambda s, token, write_test=False, state_dir=None: checked.append(token) or
                        ([] if token == "github_pat_good123" else ["the token cannot read me/gpuq (HTTP 404)"]))
    monkeypatch.setattr(setup_wizard, "check_kaggle", lambda token: (token == "KGAT_good", "GPU 30.00h"))
    monkeypatch.setattr(setup_wizard, "check_colab", lambda interactive: (True, ""))
    monkeypatch.setattr(setup_wizard, "check_hf", lambda token: (True, "Hugging Face user me"))
    monkeypatch.setattr(setup_wizard.notify, "send", lambda *a, **k: True)
    published = []
    monkeypatch.setattr(setup_wizard, "check_kaggle_dataset", lambda user, dataset, token: (True, "private"))
    monkeypatch.setattr(setup_wizard.kaggle, "publish_credentials",
                        lambda creds, user, dataset, env=None: published.append((creds, user, dataset)) or (0, "ready"))
    io = Script(answers=["y",            # github: try another token after the bad one
                         "n",            # colab: already signed in, keep
                         "y",            # huggingface: add a token
                         "y", "WANDB_API_KEY", "n",   # env: add one, then stop
                         "y"],           # ntfy: set up
                secrets_=["github_pat_bad000", "github_pat_good123", "KGAT_good", "hf_token123", "wandb123456"])
    assert setup_wizard.run(s, tmp_path, io=io, secrets_dir=d) == 0
    creds = secrets.load_credentials(d)
    assert creds == {"GITHUB_TOKEN": "github_pat_good123", "HF_TOKEN": "hf_token123", "env": {"WANDB_API_KEY": "wandb123456"}}
    assert secrets.read("kaggle_token", d) == "KGAT_good" and secrets.read("ntfy_topic", d).startswith("gpuq-")
    assert published and published[0][1:] == ("me", "gpuq-credentials") and published[0][0]["HF_TOKEN"] == "hf_token123"
    text = "\n".join(io.lines)
    assert "problem: the token cannot read me/gpuq" in text
    assert not any(v in text for v in ("github_pat_good123", "KGAT_good", "hf_token123", "wandb123456"))
    assert checked == ["github_pat_bad000", "github_pat_good123"]


def test_setup_refuses_a_broken_gpuq_toml(tmp_path):
    s = make_settings(tmp_path / "x", tmp_path / "y", tmp_path / "z")
    s["github"]["ops_repo"] = ""
    io = Script([], [])
    assert setup_wizard.run(s, tmp_path, io=io, secrets_dir=tmp_path / "s") == 2
    assert any("ops_repo is empty" in line for line in io.lines)


def test_upgrade_takes_the_new_version_and_keeps_gpuq_toml(clone, tmp_path, capsys):
    work = tmp_path / "laptop"
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "my settings", cwd=work)
    mine = (work / "gpuq.toml").read_text()
    upstream = bare_repo(tmp_path / "upstream.git", {"gpuq.toml": "upstream placeholder\n", "gpuq/new.py": "v2\n",
                                                     "README.md": "new readme\n"})
    with pytest.raises(SystemExit):  # an uncommitted change stops it
        (work / "gpuq.toml").write_text(mine + "# edit\n")
        cli.main(["upgrade", "--upstream", str(upstream)])
    git("checkout", "--", "gpuq.toml", cwd=work)
    git("config", "user.name", "t", cwd=work)
    git("config", "user.email", "t@t", cwd=work)
    capsys.readouterr()
    cli.main(["upgrade", "--upstream", str(upstream)])
    assert "committed the upgrade" in capsys.readouterr().out
    assert (work / "gpuq" / "new.py").read_text() == "v2\n" and (work / "README.md").read_text() == "new readme\n"
    assert (work / "gpuq.toml").read_text() == mine
    assert "gpuq: upgrade to" in git("log", "-1", "--format=%s", cwd=work)
    cli.main(["upgrade", "--upstream", str(upstream)])
    assert "already up to date" in capsys.readouterr().out


def test_the_credentials_dataset_must_exist_and_not_be_public(monkeypatch):
    class R:
        def __init__(self, rc, out):
            self.returncode, self.stdout = rc, out
    answers = {"status": R(0, "ready"), "http": 403}
    monkeypatch.setattr(setup_wizard.subprocess, "run", lambda *a, **k: answers["status"])
    monkeypatch.setattr(setup_wizard, "http_status", lambda url: answers["http"])
    assert setup_wizard.check_kaggle_dataset("me", "gpuq-credentials", "KGAT_x") == (True, "me/gpuq-credentials is private")
    answers["http"] = 200
    ok, detail = setup_wizard.check_kaggle_dataset("me", "gpuq-credentials", "KGAT_x")
    assert not ok and "PUBLIC" in detail
    answers["http"] = 0
    assert setup_wizard.check_kaggle_dataset("me", "gpuq-credentials", "KGAT_x")[0] is False
    answers["status"] = R(1, "404 not found")
    ok, detail = setup_wizard.check_kaggle_dataset("me", "gpuq-credentials", "KGAT_x")
    assert not ok and "does not exist yet" in detail
    assert setup_wizard.check_kaggle_dataset("", "gpuq-credentials", "KGAT_x")[0] is False
