"""Entry point of the gpuq container: refresh /data/repo to your ops repo's branch, then run the dispatcher.

The dispatcher exits when that branch moves, and the container's restart policy runs this again, so a
change you push (to gpuq.toml or to the code) reaches the dispatcher without anyone logging in.
Until `gpuq setup` has saved a GitHub token this waits and says so in `docker logs gpuq`.
"""
import base64
import json
import os
import subprocess
import sys
import time

REPO = os.environ.get("GPUQ_OPS_REPO", "")
BRANCH = os.environ.get("GPUQ_BRANCH", "main")
DIR = "/data/repo"
CREDS = os.path.join(os.environ.get("GPUQ_SECRETS", "/secrets"), "credentials.json")


def token():
    try:
        with open(CREDS, encoding="utf-8") as fh:
            return json.load(fh).get("GITHUB_TOKEN")
    except (OSError, ValueError):
        return None


if not REPO:
    sys.exit("GPUQ_OPS_REPO is not set. Start the container with install.sh")
while not token():
    print(f"no GitHub token in {CREDS} yet. Run: docker exec -it gpuq gpuq setup", flush=True)
    time.sleep(60)
url = REPO if "://" in REPO else f"https://github.com/{REPO}.git"
basic = base64.b64encode(f"x-access-token:{token()}".encode()).decode()
env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1",
       "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
       "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"}
if not os.path.isdir(os.path.join(DIR, ".git")):
    subprocess.run(["git", "init", "-q", DIR], check=True)
    subprocess.run(["git", "-C", DIR, "remote", "add", "origin", url], check=True)
for attempt in range(10):
    if subprocess.run(["git", "-C", DIR, "fetch", "-q", "--depth", "1", "origin", BRANCH], env=env).returncode == 0:
        break
    print(f"cannot fetch {REPO} {BRANCH}, retrying in a minute", flush=True)
    time.sleep(60)
else:
    sys.exit(f"cannot fetch {REPO} {BRANCH}. Check the token with: docker exec -it gpuq gpuq doctor")
subprocess.run(["git", "-C", DIR, "reset", "-q", "--hard", "FETCH_HEAD"], check=True)
os.environ.update(GPUQ_REPO_DIR=DIR, PYTHONPATH=DIR)
os.execv(sys.executable, [sys.executable, "-u", "-m", "gpuq", "dispatcher"])
