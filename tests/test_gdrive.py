"""gdrive.DriveRemote against an in-memory Drive API, and how Remote.from_env() chooses it on Colab."""
import hashlib
import io
import json
import re
import urllib.parse

import pytest

from gpuq import gdrive, storage

SECRET = "s3cr3t-value"


class FakeDrive:
    """In-memory Drive API v3: the calls gdrive makes, and nothing else."""

    def __init__(self, page=2):
        self.items = {"root": {"id": "root", "name": "", "mimeType": gdrive.FOLDER, "parents": []}}
        self.sessions, self.n, self.page = {}, 0, page
        self.fail = []  # statuses to answer, once each, before the next API call

    def _id(self):
        self.n += 1
        return f"id{self.n}"

    def _meta(self, it):
        out = {k: it[k] for k in ("id", "name", "mimeType")}
        if it["mimeType"] != gdrive.FOLDER:
            out.update(size=str(len(it["data"])), md5Checksum=hashlib.md5(it["data"]).hexdigest())
        return out

    @staticmethod
    def _ok(obj, status=200, headers=None):
        return status, headers or {}, io.BytesIO(json.dumps(obj).encode() if obj is not None else b"")

    def __call__(self, method, url, headers, body):
        data = body.read() if hasattr(body, "read") else body
        if url == gdrive.TOKEN_URL:
            form = urllib.parse.parse_qs(data.decode())
            assert form["client_secret"] == [SECRET] and form["refresh_token"] == ["r"]
            return self._ok({"access_token": "fresh", "expires_in": 3600})
        if self.fail:
            return self.fail.pop(0), {}, io.BytesIO(b"{}")
        if headers.get("Authorization") != "Bearer fresh":
            return 401, {}, io.BytesIO(b"{}")
        u = urllib.parse.urlparse(url)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.netloc == "upload.fake":
            meta = self.sessions.pop(u.path)
            fid = meta.pop("id", None) or self._id()
            it = self.items.setdefault(fid, {"id": fid, "mimeType": "application/octet-stream"})
            it.update(meta, data=data)
            return self._ok(self._meta(it))
        if u.path.startswith("/upload/drive/v3/files"):
            meta = json.loads(data or b"{}")
            if method == "PATCH":
                meta["id"] = u.path.rsplit("/", 1)[-1]
            sid = f"/s{len(self.sessions)}-{self.n}"
            self.sessions[sid] = meta
            return self._ok(None, headers={"Location": f"https://upload.fake{sid}"})
        if u.path == "/drive/v3/files" and method == "GET":
            parent = re.match(r"'([^']+)' in parents", qs["q"]).group(1)
            name = re.search(r"name = '((?:[^'\\]|\\.)*)'", qs["q"])
            want = name.group(1).replace("\\'", "'").replace("\\\\", "\\") if name else None
            hits = [it for it in self.items.values()
                    if parent in it.get("parents", []) and (want is None or it["name"] == want)]
            start = int(qs.get("pageToken", 0))
            size = 10 if name else self.page
            out = {"files": [self._meta(it) for it in hits[start:start + size]]}
            if not name and start + size < len(hits):
                out["nextPageToken"] = str(start + size)
            return self._ok(out)
        if u.path == "/drive/v3/files" and method == "POST":
            fid = self._id()
            self.items[fid] = {"id": fid, **json.loads(data)}
            return self._ok(self._meta(self.items[fid]))
        fid = u.path.rsplit("/", 1)[-1]
        if fid not in self.items:
            return 404, {}, io.BytesIO(b"{}")
        it = self.items[fid]
        if method == "GET" and qs.get("alt") == "media":
            return 200, {}, io.BytesIO(it["data"])
        if method == "DELETE":
            del self.items[fid]
            return self._ok(None, status=204)
        if method == "PATCH":
            it.update(json.loads(data))
            if "addParents" in qs:
                it["parents"] = [qs["addParents"]]
            return self._ok(self._meta(it))
        raise AssertionError(f"unexpected call {method} {url}")


def conf(tmp_path, **over):
    sec = {"type": "drive", "scope": "drive.file", "client_id": "cid", "client_secret": SECRET,
           "token": json.dumps({"access_token": "stale", "refresh_token": "r"}), **over}
    p = tmp_path / "rclone.conf"
    p.write_text("[gd]\n" + "".join(f"{k} = {v}\n" for k, v in sec.items() if v is not None), encoding="utf-8")
    return p


@pytest.fixture
def world(tmp_path):
    fake = FakeDrive()
    remote = gdrive.DriveRemote("gd:gpuq", config=conf(tmp_path), http=fake, sleep=lambda s: None)
    remote.backoff_s = 0
    return fake, remote, tmp_path


def test_put_get_ls_stat_delete_and_checkpoints(world):
    fake, remote, tmp = world
    src = tmp / "a.bin"
    src.write_bytes(b"x" * 1000)
    info = remote.put(src, "out/run/a.bin")
    assert info == {"size": 1000, "md5": hashlib.md5(b"x" * 1000).hexdigest()}
    assert remote.stat("out/run/a.bin") == info and not remote.exists("out/run/a.bin.part")   # renamed
    assert remote.get("out/run/a.bin", tmp / "back.bin").read_bytes() == b"x" * 1000
    remote.put_json({"k": 1}, "out/k.json")
    assert remote.get_json("out/k.json") == {"k": 1} and remote.get_json("out/none.json") is None
    for i in range(5):                                       # the fake answers 2 files per page
        (tmp / f"p{i}").write_text(str(i))
        remote.put(tmp / f"p{i}", f"out/parts/p{i}.txt")
    assert list(remote.ls("out/parts")) == [f"p{i}.txt" for i in range(5)] and remote.ls("nothing") == {}
    assert list(remote.ls("out")) == ["k.json"]              # folders are not listed
    src.write_bytes(b"y" * 10)                               # same name: replaced, not duplicated
    remote.put(src, "out/run/a.bin")
    assert remote.stat("out/run/a.bin")["size"] == 10 and len(remote.ls("out/run")) == 1
    remote.delete("out/run/a.bin")
    assert remote.ls("out/run") == {}
    with pytest.raises(FileNotFoundError):
        remote.get("out/run/a.bin", tmp / "x")
    ck = storage.Checkpoints(remote, "ckpt/run", keep=2)
    for step in (100, 200, 300):
        src.write_text(str(step))
        ck.save(step, src)
    assert ck.steps() == [200, 300] and ck.load(tmp / "c.pt") == 300 and (tmp / "c.pt").read_text() == "300"


def test_retries_rate_limits_and_refreshes_an_expired_token(world):
    fake, remote, tmp = world
    (tmp / "f").write_text("hola")
    remote.put(tmp / "f", "f.txt")
    fake.fail = [429, 503]
    assert remote.stat("f.txt")["size"] == 4
    remote.api.access_token = "expired"                      # the API answers 401: refresh and again
    assert remote.stat("f.txt")["size"] == 4 and remote.api.access_token == "fresh"
    fake.fail = [400]
    with pytest.raises(storage.StorageError) as e:
        remote.ls("")
    assert "HTTP 400" in str(e.value) and SECRET not in str(e.value) and "fresh" not in str(e.value)


def test_from_env_takes_the_drive_api_on_colab_only(tmp_path, capsys):
    env = {"GPUQ_REMOTE": "gd:gpuq", "RCLONE_CONFIG": str(conf(tmp_path))}
    assert isinstance(storage.Remote.from_env({**env, "GPUQ_PLATFORM": "colab"}), gdrive.DriveRemote)
    assert isinstance(storage.Remote.from_env({**env, "GPUQ_PLATFORM": "kaggle"}), storage.RcloneRemote)
    assert isinstance(storage.Remote.from_env(env), storage.RcloneRemote)                  # a laptop
    assert isinstance(storage.Remote.from_env({**env, "GPUQ_DRIVE_API": "1"}), gdrive.DriveRemote)
    assert isinstance(storage.Remote.from_env({**env, "GPUQ_PLATFORM": "colab", "GPUQ_DRIVE_API": "0"}),
                      storage.RcloneRemote)
    assert isinstance(storage.Remote.open("gd:x", config=conf(tmp_path), env={}, drive_api=True), gdrive.DriveRemote)
    capsys.readouterr()
    shared = {"GPUQ_REMOTE": "gd:gpuq", "RCLONE_CONFIG": str(conf(tmp_path, client_id=None)), "GPUQ_PLATFORM": "colab"}
    assert isinstance(storage.Remote.from_env(shared), storage.RcloneRemote)                # rclone's shared client
    err = capsys.readouterr().err
    assert "Colab reclaims VMs that run rclone" in err and SECRET not in err
    with pytest.raises(gdrive.DriveError):
        gdrive.read_remote(conf(tmp_path, type="s3"), "gd")
    assert gdrive.rclone_config_path(None, {"RCLONE_CONFIG": "/x/r.conf"}).as_posix() == "/x/r.conf"
