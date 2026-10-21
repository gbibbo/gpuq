"""Secrets of the dispatcher machine. They never leave it except in two ways: the credentials bundle
is uploaded to each Colab VM for one attempt (and deleted there after reading), and the same bundle
is stored in a private Kaggle dataset that Kaggle runs read.

Folder layout (GPUQ_SECRETS, /secrets inside the container, mode 700, files mode 600):

    credentials.json   {"GITHUB_TOKEN": ..., "RCLONE_CONF": ..., "HF_TOKEN": ..., "env": {"WANDB_API_KEY": ...}}
    kaggle_token       Kaggle API token, used only by the dispatcher (never sent to a VM)
    ntfy_topic         private ntfy topic for notifications (optional)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

CREDENTIALS = "credentials.json"
KAGGLE_TOKEN = "kaggle_token"
NTFY_TOPIC = "ntfy_topic"


def folder(path=None) -> Path:
    return Path(path or os.environ.get("GPUQ_SECRETS") or "/secrets")


def _write_private(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def load_credentials(d=None) -> dict:
    p = folder(d) / CREDENTIALS
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_credentials(creds: dict, d=None):
    clean = {k: v for k, v in creds.items() if v not in (None, "", {})}
    _write_private(folder(d) / CREDENTIALS, json.dumps(clean, indent=2))


def read(name: str, d=None):
    p = folder(d) / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def write(name: str, value: str, d=None):
    _write_private(folder(d) / name, value.strip() + "\n")


def fragments(creds: dict, *extra) -> list:
    """Strings to hide in anything published: every credential value, the env values, and the
    tokens and client secret inside an rclone config."""
    out = []
    for v in [*creds.values(), *(creds.get("env") or {}).values(), *extra]:
        if isinstance(v, str) and len(v) >= 8:
            out.append(v)
    conf = creds.get("RCLONE_CONF") or ""
    if conf and "[" not in conf:
        try:
            import base64
            conf = base64.b64decode(conf).decode("utf-8")
        except Exception:
            conf = ""
    for line in conf.splitlines():
        key, _, value = line.partition("=")
        if key.strip() in ("token", "client_secret", "pass", "password", "secret_access_key", "key") and value.strip():
            out.append(value.strip())
    for m in re.finditer(r'"(?:access_token|refresh_token)"\s*:\s*"([^"]+)"', conf):
        out.append(m.group(1))
    return sorted({f for f in out if len(f) >= 8}, key=len, reverse=True)


def redact(text: str, frags) -> str:
    for f in frags:
        text = text.replace(f, "***")
    return text
