"""Shared helpers: bare git repos that play GitHub, and the settings of a test installation."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKES = Path(__file__).resolve().parent / "fakes"
sys.path.insert(0, str(ROOT))

from gpuq import settings as settings_mod  # noqa: E402


def git(*args, cwd=None) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr}")
    return r.stdout


def bare_repo(path: Path, files: dict, branch: str = "main") -> Path:
    """A bare repo whose `branch` holds `files` ({relpath: text or source Path})."""
    work = path.with_name(path.name + "-work")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    git("init", "-q", "-b", branch, cwd=work)
    for rel, content in files.items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):
            if content.is_dir():
                shutil.copytree(content, p, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copyfile(content, p)
        else:
            p.write_text(content, encoding="utf-8")
    git("add", "-A", cwd=work)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=work)
    git("init", "-q", "--bare", "-b", branch, str(path))
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=path)
    git("push", "-q", str(path), f"{branch}:{branch}", cwd=work)
    return path


def show(bare: Path, branch: str, rel: str):
    r = subprocess.run(["git", "--git-dir", str(bare), "show", f"{branch}:{rel}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def make_settings(ops: Path, code: Path, remote: Path, **over) -> dict:
    s = settings_mod.load(Path("/nonexistent/gpuq.toml"), env={})
    s["github"].update(ops_repo=str(ops), code_repo=str(code), ref="main")
    s["kaggle"].update(username="me")
    s["storage"]["remote"] = str(remote)
    for section, values in over.items():
        s[section].update(values)
    return s


JOB_PY = '''import os, sys, time
print("job running on", os.environ["GPUQ_PLATFORM"], "attempt", os.environ["GPUQ_ATTEMPT"], flush=True)
print("secret is", os.environ.get("MY_SECRET"), flush=True)
from gpuq.storage import Remote
Remote.from_env().put_json({"job": os.environ["GPUQ_JOB_ID"]}, "out/" + os.environ["GPUQ_JOB_ID"] + ".json")
time.sleep(float(sys.argv[sys.argv.index("--sleep") + 1]) if "--sleep" in sys.argv else 0.3)
print("job done", flush=True)
sys.exit(int(sys.argv[sys.argv.index("--rc") + 1]) if "--rc" in sys.argv else 0)
'''


@pytest.fixture
def world(tmp_path):
    """An ops repo (with the gpuq package, like a user's copy), a code repo, a remote folder and a
    secrets folder with credentials."""
    ops = bare_repo(tmp_path / "ops.git", {"gpuq": ROOT / "gpuq", "gpuq.toml": "[github]\n"})
    code = bare_repo(tmp_path / "code.git", {"job.py": JOB_PY})
    remote = tmp_path / "remote"
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    creds = {"GITHUB_TOKEN": "ghp_faketokenfortests000000", "env": {"MY_SECRET": "supersecretvalue42"}}
    (secrets_dir / "credentials.json").write_text(json.dumps(creds))
    (secrets_dir / "kaggle_token").write_text("KGAT_fakekaggletoken0000\n")
    return {"tmp": tmp_path, "ops": ops, "code": code, "remote": remote, "secrets": secrets_dir, "creds": creds,
            "settings": make_settings(ops, code, remote)}
