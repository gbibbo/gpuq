"""Settings of one gpuq installation, read from gpuq.toml at the root of your copy of the repo.

gpuq.toml holds no secrets and is committed. Secrets live only on the dispatcher machine
(gpuq.secrets). The dispatcher reads gpuq.toml from its own checkout of the repo, so a change
pushed to the default branch reaches it on its next restart.
"""
from __future__ import annotations

import copy
import os
import re
import tomllib
from pathlib import Path

FILE = "gpuq.toml"

DEFAULTS = {
    "github": {
        "ops_repo": "",       # owner/name of this repo: queue on branch jobs, status and logs on branch runs
        "branch": "main",     # the dispatcher runs this branch of the ops repo and restarts when it moves
        "code_repo": "",      # repo that jobs check out unless they name another (default: the ops repo)
        "ref": "main",        # branch, tag or commit of the code repo (default for jobs)
    },
    "colab": {
        "enabled": True,
        "session": "gpuq",    # name of the Colab CLI session
        "cpu_jobs": False,    # CPU-only Colab VMs have died about 12 min after assignment (docs/HOW_IT_WORKS.md)
    },
    "kaggle": {
        "enabled": True,
        "username": "",
        "kernel": "gpuq-launcher",     # one private notebook, reused for every Kaggle job
        "dataset": "gpuq-credentials",  # private dataset that carries the credentials to Kaggle runs
    },
    "storage": {
        "remote": "",         # rclone remote:path that jobs receive as GPUQ_REMOTE, e.g. gdrive:gpuq
    },
    "jobs": {
        "gpu": "T4",               # default accelerator of a job. T4 is Colab's free one, CPU for none
        "max_failures": 2,         # attempts that end in an error of the job or of its setup
        "max_interruptions": 10,   # attempts cut by the platform; each one is relaunched
        "timeout_h": 12,
    },
    "dispatcher": {
        "tick_s": 60,
        "poll_s": 60,
        "log_publish_s": 600,
        "log_tail_kb": 64,
    },
}

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class SettingsError(ValueError):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path=None, env=None) -> dict:
    """gpuq.toml merged over the defaults. GPUQ_OPS_REPO in the environment overrides github.ops_repo."""
    env = os.environ if env is None else env
    path = Path(path or FILE)
    raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    s = _merge(DEFAULTS, raw)
    if env.get("GPUQ_OPS_REPO"):
        s["github"]["ops_repo"] = env["GPUQ_OPS_REPO"]
    s["github"]["code_repo"] = s["github"]["code_repo"] or s["github"]["ops_repo"]
    return s


def check(s: dict, need_kaggle_user: bool = True) -> list:
    """Problems that stop the dispatcher, as sentences."""
    problems = []
    ops = s["github"]["ops_repo"]
    if not ops:
        problems.append("github.ops_repo is empty in gpuq.toml")
    elif not (REPO_RE.match(ops) or "://" in ops or os.path.isabs(ops) or Path(ops).exists()):
        problems.append(f"github.ops_repo {ops!r} is not owner/name")
    if s["kaggle"]["enabled"] and need_kaggle_user and not s["kaggle"]["username"]:
        problems.append("kaggle.enabled is true but kaggle.username is empty in gpuq.toml")
    if not (s["colab"]["enabled"] or s["kaggle"]["enabled"]):
        problems.append("both colab.enabled and kaggle.enabled are false")
    return problems

