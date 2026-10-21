"""Reads and writes one branch of the ops repo: jobs (the queue) or runs (status and logs).

Every write fetches the tip of the branch, resets a private shallow clone to it, applies the change,
commits and pushes. A rejected push is retried from the fetch with growing waits, which has the
effect of pull --rebase. A branch that does not exist yet is created as an orphan.

With a token (the dispatcher), the token travels in an HTTP header set through GIT_CONFIG_*
variables and never appears in a command line, a remote URL or .git/config. Without a token (your
laptop), git uses your usual credentials for the repo URL.
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

BOT = ("gpuq-bot", "gpuq-bot@users.noreply.github.com")


class GitError(RuntimeError):
    pass


def repo_url(repo: str) -> str:
    """'owner/name' to an https URL. Full URLs and local paths pass through."""
    if "://" in repo or repo.startswith(("/", "git@")) or os.path.isabs(repo) or Path(repo).exists():
        return repo
    return f"https://github.com/{repo}.git"


def auth_env(token, url: str) -> dict:
    env = {"GIT_TERMINAL_PROMPT": "0"}
    if token and url.startswith("https://"):
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        host = "/".join(url.split("/")[:3]) + "/"
        env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"http.{host}.extraheader",
                    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"})
    return env


class Branch:
    def __init__(self, repo: str, branch: str, clone_dir, token=None, author=BOT, retries: int = 6,
                 backoff_s: float = 2.0, redact=()):
        self.url = repo_url(repo)
        self.branch = branch
        self.dir = Path(clone_dir)
        self.author = author
        self.retries = retries
        self.backoff_s = backoff_s
        self.secrets = [s for s in (token, *redact) if s]
        self._lock = threading.Lock()
        self._env = {**os.environ, **auth_env(token, self.url)}
        if not token:
            self._env.pop("GIT_TERMINAL_PROMPT")  # let a credential helper ask on a laptop

    def _clean(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "***")
        return text

    def _git(self, *args, check: bool = True):
        res = subprocess.run(["git", *args], cwd=str(self.dir), env=self._env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        if check and res.returncode != 0:
            raise GitError(self._clean(f"git {args[0]} exited {res.returncode}: {res.stderr.strip()[-800:]}"))
        return res

    def _ensure_clone(self):
        if (self.dir / ".git").is_dir():
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q")
        self._git("config", "core.autocrlf", "false")  # files exactly as published, also on Windows
        self._git("remote", "add", "origin", self.url)

    def _sync(self) -> bool:
        """Working tree at the remote tip. False when the branch does not exist on the remote."""
        res = self._git("fetch", "-q", "--depth", "1", "origin",
                        f"+refs/heads/{self.branch}:refs/remotes/origin/{self.branch}", check=False)
        if res.returncode != 0:
            err = res.stderr.lower()
            if "couldn't find remote ref" in err or "could not find remote ref" in err:
                return False
            raise GitError(self._clean(f"git fetch exited {res.returncode}: {res.stderr.strip()[-800:]}"))
        self._git("checkout", "-q", "-B", self.branch, f"origin/{self.branch}")
        self._git("reset", "-q", "--hard", f"origin/{self.branch}")
        self._git("clean", "-q", "-fdx")
        return True

    def _orphan(self):
        self._git("checkout", "-q", "--orphan", f"gpuq-new-{self.branch}")
        self._git("rm", "-q", "-rf", "--ignore-unmatch", ".")
        self._git("clean", "-q", "-fdx")
        (self.dir / "README.md").write_bytes(f"gpuq branch `{self.branch}`. Written by gpuq, do not edit by hand.\n"
                                             .encode("utf-8"))

    def read(self, rel: str):
        """Bytes of a file at the remote tip, or None."""
        with self._lock:
            self._ensure_clone()
            if not self._sync():
                return None
            p = self.dir / rel
            return p.read_bytes() if p.is_file() else None

    def head(self):
        """Commit at the remote tip, or None."""
        with self._lock:
            self._ensure_clone()
            if not self._sync():
                return None
            return self._git("rev-parse", "HEAD").stdout.strip()

    def update(self, change, message: str) -> bool:
        """change(root: Path) edits files under the clone. Commits and pushes. Returns False when the
        change left nothing to commit."""
        with self._lock:
            self._ensure_clone()
            last = None
            for attempt in range(self.retries):
                try:
                    if not self._sync():
                        self._orphan()
                    change(self.dir)
                    self._git("add", "-A")
                    if self._git("diff", "--cached", "--quiet", check=False).returncode == 0:
                        return False
                    self._git("-c", f"user.name={self.author[0]}", "-c", f"user.email={self.author[1]}",
                              "commit", "-q", "-m", message)
                    self._git("push", "-q", "origin", f"HEAD:refs/heads/{self.branch}")
                    return True
                except GitError as exc:
                    last = exc
                    time.sleep(self.backoff_s * (2 ** attempt))
            raise GitError(f"writing branch {self.branch} failed after {self.retries} attempts: {last}")

    def publish(self, files: dict, delete=(), message: str = "gpuq: update") -> bool:
        """Write {relpath: str | bytes | Path} and delete paths, in one commit."""
        def change(root: Path):
            for rel, content in files.items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, Path):
                    p.write_bytes(content.read_bytes())
                elif isinstance(content, bytes):
                    p.write_bytes(content)
                else:
                    p.write_bytes(str(content).encode("utf-8"))  # no newline translation on Windows
            for rel in delete:
                p = root / rel
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink(missing_ok=True)
        return self.update(change, message)
