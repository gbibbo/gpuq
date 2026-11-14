"""Google Drive through its HTTP API, from Python, for jobs on Colab. Standard library only.

Since November 2026 Colab reclaims, about 12 minutes after assignment, every VM that runs the rclone
program, whatever rclone connects to (docs/HOW_IT_WORKS.md, Known issues). DriveRemote is a storage.Remote
on top of the Drive API v3 instead: the same atomic, verified put, get, ls, stat, exists and delete. It
authenticates with the OAuth client and token of the rclone remote, so it is the same application as rclone
and sees the same files, also under the drive.file scope. It keeps the refreshed access token in memory,
never writes rclone.conf and never puts a credential in an error or a log.

Remote.from_env() and Remote.open() choose it by themselves on Colab (GPUQ_PLATFORM=colab) when the
remote is a Google Drive remote with its own client_id; GPUQ_DRIVE_API=1 forces it anywhere and =0 forbids it.
"""
from __future__ import annotations

import configparser
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .storage import Remote, StorageError

API = "https://www.googleapis.com/drive/v3"
UPLOAD = "https://www.googleapis.com/upload/drive/v3"
TOKEN_URL = "https://oauth2.googleapis.com/token"
FOLDER = "application/vnd.google-apps.folder"
FIELDS = "id,name,mimeType,size,md5Checksum"
CHUNK = 1 << 20
RETRY_STATUS = (429, 500, 502, 503, 504)


class DriveError(StorageError):
    """An answer of the Drive API that is not a missing file. Never carries a credential."""

    def __init__(self, status: int, what: str):
        super().__init__(f"Drive API {what}: HTTP {status}" if status else f"Drive API: {what}")
        self.status = status


def rclone_config_path(config=None, env=None) -> Path:
    """rclone.conf: the given path, RCLONE_CONFIG, else rclone's default place on this system."""
    env = os.environ if env is None else env
    if config or env.get("RCLONE_CONFIG"):
        return Path(config or env["RCLONE_CONFIG"])
    if os.name == "nt" and env.get("APPDATA"):
        return Path(env["APPDATA"]) / "rclone" / "rclone.conf"
    return Path(env.get("HOME") or Path.home()) / ".config" / "rclone" / "rclone.conf"


def read_remote(config_path, name: str) -> dict:
    """The [name] section of rclone.conf, checked to be a Google Drive remote with its own OAuth client."""
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(config_path, encoding="utf-8")
    if not cp.has_section(name):
        raise DriveError(0, f"no remote [{name}] in the rclone config")
    sec = dict(cp.items(name))
    if sec.get("type") != "drive":
        raise DriveError(0, f"remote [{name}] is not a Google Drive remote")
    if not (sec.get("client_id") and sec.get("client_secret") and sec.get("token")):
        raise DriveError(0, f"remote [{name}] has no client_id, client_secret or token of its own")
    return sec


def use_api(base: str, config=None, env=None) -> bool:
    """Whether a remote 'name:folder' goes through the Drive API instead of rclone. GPUQ_DRIVE_API=1 or 0
    decides; otherwise only on Colab, and only for a Drive remote with its own OAuth client, with a warning on
    stderr when Colab would have to run rclone."""
    env = os.environ if env is None else env
    forced = env.get("GPUQ_DRIVE_API")
    if forced in ("0", "1"):
        return forced == "1"
    if env.get("GPUQ_PLATFORM") != "colab":
        return False
    try:
        read_remote(rclone_config_path(config, env), base.split(":", 1)[0])
        return True
    except DriveError as exc:
        print(f"gpuq: {exc}, so this job reaches it with rclone, and Colab reclaims VMs that run rclone about "
              "12 minutes after assignment (docs/HOW_IT_WORKS.md, Known issues)", file=sys.stderr, flush=True)
        return False


def _urllib_http(method, url, headers, body):
    """(status, headers, response) for one request. The response is a file-like object to read."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        r = urllib.request.urlopen(req, timeout=300)
        return r.status, dict(r.headers), r
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e


class DriveApi:
    """Authenticated requests to the Drive API, with token refresh and retries."""

    def __init__(self, remote: dict, http=_urllib_http, sleep=time.sleep, retries: int = 5):
        self.client_id, self.client_secret = remote["client_id"], remote["client_secret"]
        tok = json.loads(remote["token"])
        self.refresh_token = tok.get("refresh_token")
        self.access_token = tok.get("access_token")
        self.expires_at = 0.0  # refresh before the first call: the stored expiry may be stale
        self.http, self.sleep, self.retries = http, sleep, retries

    def _refresh(self):
        body = urllib.parse.urlencode({"client_id": self.client_id, "client_secret": self.client_secret,
                                       "refresh_token": self.refresh_token,
                                       "grant_type": "refresh_token"}).encode()
        status, _, r = self.http("POST", TOKEN_URL, {"Content-Type": "application/x-www-form-urlencoded"}, body)
        data = r.read()
        if status != 200:
            raise DriveError(status, "token refresh")
        tok = json.loads(data)
        self.access_token = tok["access_token"]
        self.expires_at = time.time() + int(tok.get("expires_in", 3600)) - 120

    def request(self, method: str, url: str, what: str, headers=None, body=None, ok=(200,), missing=(404,)):
        """(status, headers, response) of a successful call, None if missing; retries 401, 403, 429 and 5xx.
        body may be bytes or a callable returning a fresh file object, so a retry can send it again."""
        status = 0
        for attempt in range(self.retries):
            if time.time() >= self.expires_at:
                self._refresh()
            h = {"Authorization": f"Bearer {self.access_token}", **(headers or {})}
            data = body() if callable(body) else body
            try:
                status, rh, r = self.http(method, url, h, data)
            finally:
                if hasattr(data, "close"):
                    data.close()
            if status in ok:
                return status, rh, r
            r.read()
            if status in missing:
                return None
            if status == 401:
                self.expires_at = 0.0
            elif status not in RETRY_STATUS and not (status == 403 and attempt < self.retries - 1):
                raise DriveError(status, what)
            self.sleep(min(60, 2 ** attempt))
        raise DriveError(status, what)

    def json(self, method, url, what, payload=None, **kw):
        headers = {"Content-Type": "application/json"} if payload is not None else None
        body = json.dumps(payload).encode() if payload is not None else None
        res = self.request(method, url, what, headers=headers, body=body, **kw)
        return None if res is None else json.loads(res[2].read() or b"{}")


def _q(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


class DriveRemote(Remote):
    """A Google Drive remote 'name:folder' of rclone.conf, reached through the Drive API."""

    def __init__(self, base: str, config=None, http=_urllib_http, sleep=time.sleep, env=None):
        self.base = base.rstrip("/")
        name, _, folder = self.base.partition(":")
        sec = read_remote(rclone_config_path(config, env), name)
        self.api = DriveApi(sec, http=http, sleep=sleep)
        self.folder = folder.strip("/")
        self.folders = {"": sec.get("root_folder_id") or "root"}

    def _full(self, rel: str) -> str:
        return "/".join(p for p in (self.folder, rel.strip("/")) if p)

    def _child(self, parent: str, name: str):
        q = f"'{parent}' in parents and name = '{_q(name)}' and trashed = false"
        url = f"{API}/files?" + urllib.parse.urlencode({"q": q, "fields": f"files({FIELDS})",
                                                        "orderBy": "modifiedTime desc", "pageSize": 10})
        files = self.api.json("GET", url, f"lookup of {name}")["files"]
        return files[0] if files else None

    def _folder(self, path: str, create: bool = False):
        if path in self.folders:
            return self.folders[path]
        head, _, last = path.rpartition("/")
        parent = self._folder(head, create)
        if parent is None:
            return None
        item = self._child(parent, last)
        if item is not None and item["mimeType"] != FOLDER:
            raise StorageError(f"{path} is a file, not a folder")
        if item is None:
            if not create:
                return None
            item = self.api.json("POST", f"{API}/files?fields={FIELDS}", f"folder {last}",
                                 {"name": last, "mimeType": FOLDER, "parents": [parent]})
        self.folders[path] = item["id"]
        return item["id"]

    def _file(self, rel: str):
        head, _, last = self._full(rel).rpartition("/")
        parent = self._folder(head)
        item = self._child(parent, last) if parent else None
        return None if item is None or item["mimeType"] == FOLDER else item

    @staticmethod
    def _entry(item) -> dict:
        return {"size": int(item.get("size", 0)), "md5": item.get("md5Checksum") or None}

    @staticmethod
    def _guard(what, fn):
        try:
            return fn()
        except StorageError:
            raise
        except (OSError, ValueError) as exc:  # network errors, broken answers: a retry of put/get may fix them
            raise StorageError(f"Drive API {what}: {type(exc).__name__}") from None

    def _stat(self, rel):
        def run():
            item = self._file(rel)
            return self._entry(item) if item else None
        return self._guard(f"stat {rel}", run)

    def _list(self, rel):
        def run():
            fid = self._folder(self._full(rel))
            if fid is None:
                return {}
            out, token = {}, None
            while True:
                params = {"q": f"'{fid}' in parents and trashed = false", "pageSize": 1000,
                          "fields": f"nextPageToken,files({FIELDS})"}
                if token:
                    params["pageToken"] = token
                page = self.api.json("GET", f"{API}/files?" + urllib.parse.urlencode(params), f"list {rel}")
                for it in page["files"]:
                    if it["mimeType"] != FOLDER:
                        out.setdefault(it["name"], self._entry(it))
                token = page.get("nextPageToken")
                if not token:
                    return dict(sorted(out.items()))
        return self._guard(f"list {rel}", run)

    def _download(self, rel, local):
        def run():
            item = self._file(rel)
            if item is None:
                raise StorageError(f"{rel} missing")
            res = self.api.request("GET", f"{API}/files/{item['id']}?alt=media", f"download {rel}")
            with open(local, "wb") as fh:
                while True:
                    block = res[2].read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
        return self._guard(f"download {rel}", run)

    def _upload(self, local, rel):
        def run():
            head, _, last = self._full(rel).rpartition("/")
            parent = self._folder(head, create=True)
            old = self._child(parent, last)
            size = Path(local).stat().st_size
            if old is None:
                method, url, meta = "POST", f"{UPLOAD}/files?uploadType=resumable", {"name": last, "parents": [parent]}
            else:
                method, url, meta = "PATCH", f"{UPLOAD}/files/{old['id']}?uploadType=resumable", {}
            res = self.api.request(method, url, f"upload start {rel}", body=json.dumps(meta).encode(),
                                   headers={"Content-Type": "application/json; charset=UTF-8",
                                            "X-Upload-Content-Length": str(size)})
            location = {k.lower(): v for k, v in res[1].items()}["location"]
            res[2].read()
            self.api.request("PUT", location, f"upload {rel}", body=lambda: open(local, "rb"),
                             headers={"Content-Length": str(size)}, ok=(200, 201))[2].read()
        return self._guard(f"upload {rel}", run)

    def _move(self, src, dst):
        def run():
            item = self._file(src)
            if item is None:
                raise StorageError(f"{src} missing before rename")
            old = self._file(dst)
            if old is not None and old["id"] != item["id"]:
                self.api.request("DELETE", f"{API}/files/{old['id']}", f"delete {dst}", ok=(200, 204))
            s_head = self._full(src).rpartition("/")[0]
            d_head, _, d_last = self._full(dst).rpartition("/")
            url = f"{API}/files/{item['id']}?fields={FIELDS}"
            if d_head != s_head:
                url += "&" + urllib.parse.urlencode({"addParents": self._folder(d_head, create=True),
                                                     "removeParents": self._folder(s_head)})
            self.api.json("PATCH", url, f"rename {src}", {"name": d_last})
        return self._guard(f"move {src}", run)

    def _delete(self, rel):
        def run():
            item = self._file(rel)
            if item is not None:
                self.api.request("DELETE", f"{API}/files/{item['id']}", f"delete {rel}", ok=(200, 204))
        return self._guard(f"delete {rel}", run)
