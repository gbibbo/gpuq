"""Remote storage for your jobs, through rclone. Standard library only.

A VM can disappear at any moment and its disk goes with it. Whatever must survive an interruption
(checkpoints, finished outputs) goes to the remote the job receives as GPUQ_REMOTE, for example
`gdrive:gpuq`. The write protocol makes a half-written file impossible to mistake for a good one.
put() uploads to <name>.part, renames it to <name> and checks size and MD5 of the remote file
against the local one. A file counts as stored only after put() returns.

    from gpuq.storage import Remote, Checkpoints
    remote = Remote.from_env()                      # GPUQ_REMOTE and RCLONE_CONFIG, set by gpuq
    ckpt = Checkpoints(remote, f"checkpoints/{run_name}", keep=2)
    step = ckpt.load("/tmp/ckpt.pt")                # None on the first attempt
    ...
    ckpt.save(step, "/tmp/ckpt.pt")                 # every N steps
    remote.put("results.csv", f"results/{run_name}/results.csv")

LocalRemote stores files in a local folder and stands in for the remote in tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

PART = ".part"


class StorageError(RuntimeError):
    pass


def md5_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _join(*parts) -> str:
    return "/".join(p.strip("/") for p in parts if p and p.strip("/"))


class Remote:
    """Write protocol and helpers on top of six backend primitives."""

    retries = 3
    backoff_s = 2.0

    def _upload(self, local: Path, rel: str): raise NotImplementedError
    def _download(self, rel: str, local: Path): raise NotImplementedError
    def _move(self, src: str, dst: str): raise NotImplementedError
    def _stat(self, rel: str): raise NotImplementedError
    def _list(self, rel: str) -> dict: raise NotImplementedError
    def _delete(self, rel: str): raise NotImplementedError

    @staticmethod
    def from_env(env=None) -> "Remote":
        """The remote gpuq gives the job. A plain local folder works too, for tests."""
        env = os.environ if env is None else env
        base = env.get("GPUQ_REMOTE")
        if not base:
            raise StorageError("GPUQ_REMOTE is not set. Set storage.remote in gpuq.toml and rclone in gpuq setup")
        if ":" not in base.split("/")[0] or os.path.isabs(base):
            return LocalRemote(base)
        return RcloneRemote(base, config=env.get("RCLONE_CONFIG"))

    def _retry(self, what: str, fn):
        last = None
        for attempt in range(self.retries):
            try:
                return fn()
            except StorageError as exc:
                last = exc
                time.sleep(self.backoff_s * (2 ** attempt))
        raise StorageError(f"{what} failed after {self.retries} attempts: {last}")

    def put(self, local, rel: str) -> dict:
        """Atomic, verified upload. Returns {'size', 'md5'} of the stored file."""
        local = Path(local)
        want = {"size": local.stat().st_size, "md5": md5_file(local)}

        def attempt():
            self._upload(local, rel + PART)
            self._move(rel + PART, rel)
            got = self._stat(rel)
            if got is None:
                raise StorageError(f"{rel} missing after rename")
            if got["size"] != want["size"] or (got.get("md5") and got["md5"] != want["md5"]):
                raise StorageError(f"{rel} verification failed: remote {got}, local {want}")
            return want

        return self._retry(f"put {rel}", attempt)

    def put_json(self, obj, rel: str) -> dict:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / Path(rel).name
            p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
            return self.put(p, rel)

    def get(self, rel: str, local) -> Path:
        """Download and verify against the remote listing."""
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)

        def attempt():
            info = self._stat(rel)
            if info is None:
                raise FileNotFoundError(rel)
            tmp = local.with_name(local.name + PART)
            self._download(rel, tmp)
            if tmp.stat().st_size != info["size"] or (info.get("md5") and md5_file(tmp) != info["md5"]):
                tmp.unlink(missing_ok=True)
                raise StorageError(f"{rel} download does not match the remote listing")
            tmp.replace(local)
            return local

        return self._retry(f"get {rel}", attempt)

    def get_json(self, rel: str):
        if self._stat(rel) is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            return json.loads(self.get(rel, Path(d) / "x.json").read_text(encoding="utf-8"))

    def ls(self, rel: str) -> dict:
        """{name: {'size', 'md5'}} for the files directly under rel, {} if missing."""
        return self._list(rel)

    def stat(self, rel: str):
        return self._stat(rel)

    def exists(self, rel: str) -> bool:
        return self._stat(rel) is not None

    def delete(self, rel: str):
        if self._stat(rel) is not None:
            self._delete(rel)


class LocalRemote(Remote):
    backoff_s = 0.0

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, rel):
        return self.root / rel

    def _upload(self, local, rel):
        self._p(rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, self._p(rel))

    def _download(self, rel, local):
        shutil.copyfile(self._p(rel), local)

    def _move(self, src, dst):
        self._p(dst).parent.mkdir(parents=True, exist_ok=True)
        self._p(src).replace(self._p(dst))

    def _stat(self, rel):
        p = self._p(rel)
        return {"size": p.stat().st_size, "md5": md5_file(p)} if p.is_file() else None

    def _list(self, rel):
        d = self._p(rel)
        if not d.is_dir():
            return {}
        return {p.name: {"size": p.stat().st_size, "md5": md5_file(p)} for p in sorted(d.iterdir()) if p.is_file()}

    def _delete(self, rel):
        self._p(rel).unlink(missing_ok=True)


class RcloneRemote(Remote):
    """Any rclone remote. base is 'remote:folder'."""

    def __init__(self, base: str, config=None, rclone: str = "rclone",
                 flags=("--retries", "3", "--low-level-retries", "10")):
        self.base, self.config, self.rclone, self.flags = base.rstrip("/"), config, rclone, list(flags)

    def _t(self, rel):
        return f"{self.base}/{rel}" if rel else self.base

    def _run(self, *args, check: bool = True):
        cmd = [self.rclone, *args, *self.flags] + (["--config", str(self.config)] if self.config else [])
        res = subprocess.run(cmd, capture_output=True, text=True)
        if check and res.returncode != 0:
            raise StorageError(f"rclone {args[0]} exited {res.returncode}: {res.stderr.strip()[-800:]}")
        return res

    def _upload(self, local, rel):
        self._run("copyto", str(local), self._t(rel))

    def _download(self, rel, local):
        self._run("copyto", self._t(rel), str(local))

    def _move(self, src, dst):
        self._run("moveto", self._t(src), self._t(dst))

    @staticmethod
    def _entry(e: dict) -> dict:
        hashes = {k.lower(): v for k, v in (e.get("Hashes") or {}).items()}
        return {"size": int(e["Size"]), "md5": hashes.get("md5") or None}

    def _stat(self, rel):
        res = self._run("lsjson", "--stat", "--hash", self._t(rel), check=False)
        if res.returncode != 0:
            if res.returncode in (3, 4) or "not found" in res.stderr.lower():
                return None
            raise StorageError(f"rclone lsjson --stat exited {res.returncode}: {res.stderr.strip()[-800:]}")
        e = json.loads(res.stdout)
        return None if e.get("IsDir") else self._entry(e)

    def _list(self, rel):
        res = self._run("lsjson", "--files-only", "--hash", self._t(rel), check=False)
        if res.returncode != 0:
            if res.returncode in (3, 4) or "not found" in res.stderr.lower():
                return {}
            raise StorageError(f"rclone lsjson exited {res.returncode}: {res.stderr.strip()[-800:]}")
        return {e["Name"]: self._entry(e) for e in json.loads(res.stdout or "[]")}

    def _delete(self, rel):
        self._run("deletefile", self._t(rel))


class Checkpoints:
    """The last `keep` checkpoints of one run, stored as <prefix>/step_<n><suffix>."""

    NAME = re.compile(r"^step_(\d+)(\..+)?$")

    def __init__(self, remote: Remote, prefix: str, keep: int = 2, suffix: str = ".pt"):
        self.remote, self.prefix, self.keep, self.suffix = remote, prefix.strip("/"), keep, suffix

    def steps(self) -> list:
        out = []
        for name in self.remote.ls(self.prefix):
            m = self.NAME.match(name)
            if m and not name.endswith(PART) and (m.group(2) or "") == self.suffix:
                out.append(int(m.group(1)))
        return sorted(out)

    def latest(self):
        s = self.steps()
        return s[-1] if s else None

    def _rel(self, step: int) -> str:
        return _join(self.prefix, f"step_{step:09d}{self.suffix}")

    def save(self, step: int, local) -> dict:
        info = self.remote.put(local, self._rel(step))
        for old in self.steps()[:-self.keep] if self.keep > 0 else []:
            self.remote.delete(self._rel(old))
        return info

    def load(self, local, step=None):
        """Download the latest (or the given) checkpoint to `local`. Returns its step, or None."""
        step = self.latest() if step is None else step
        if step is None:
            return None
        self.remote.get(self._rel(step), local)
        return step
