"""Kaggle side: the kernel folder that `kaggle kernels push` runs, the private credentials dataset and
the parsing of the CLI's answers.

Kaggle does not hand notebook Secrets to kernels pushed with the CLI (Kaggle CLI issue 582), so the
credentials bundle travels in a private dataset attached to the kernel. If that dataset were made
public, the tokens in it would leak, so `gpuq setup` creates it private and `gpuq doctor` checks it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import vm

SHAPES = {"T4": "NvidiaTeslaT4", "P100": "NvidiaTeslaP100"}
QUOTA_MARKERS = ("quota", "exceeded your", "weekly limit", "gpu limit", "maximum number of")


def build_kernel(folder: Path, script: str, user: str, kernel: str, dataset: str, gpu: str) -> Path:
    folder = Path(folder)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "gpuq_launcher.ipynb").write_text(vm.kaggle_notebook(script), encoding="utf-8")
    shape = SHAPES.get(gpu.upper()) if gpu.upper() != "CPU" else None
    meta = {"id": f"{user}/{kernel}", "title": kernel, "code_file": "gpuq_launcher.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": shape is not None,
            "enable_internet": True, "machine_shape": shape or "", "dataset_sources": [f"{user}/{dataset}"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def state(status_line: str) -> str:
    """'running', 'complete', 'error' or 'cancelled' from `kaggle kernels status`."""
    s = status_line.lower()
    if "complete" in s:
        return "complete"
    if "cancel" in s:
        return "cancelled"
    if "error" in s:
        return "error"
    return "running"


def log_text(raw: bytes) -> str:
    """Text of a Kaggle kernel log (a JSON list of {stream_name, data}), or the raw text."""
    txt = raw.decode("utf-8", errors="replace")
    try:
        return "".join(e.get("data", "") for e in json.loads(txt))
    except Exception:
        return txt


def publish_credentials(creds: dict, user: str, dataset: str, kaggle_cmd=("kaggle",), env=None) -> tuple:
    """Create the private dataset with gpuq_credentials.json, or add a version. Returns (rc, output)
    with every secret replaced by ***."""
    ref = f"{user}/{dataset}"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / vm.CREDENTIALS_FILE).write_text(json.dumps(creds), encoding="utf-8")
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": dataset, "id": ref, "licenses": [{"name": "other"}]}), encoding="utf-8")
        st = subprocess.run([*kaggle_cmd, "datasets", "status", ref], capture_output=True, text=True, env=env)
        if st.returncode == 0 and "ready" in st.stdout.lower():
            cmd = [*kaggle_cmd, "datasets", "version", "-p", str(d), "-m", "credentials updated"]
        else:
            cmd = [*kaggle_cmd, "datasets", "create", "-p", str(d)]  # private unless --public is passed
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    from .secrets import fragments, redact
    return res.returncode, redact((res.stdout + res.stderr).strip(), fragments(creds))
