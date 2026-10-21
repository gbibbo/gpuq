"""First job: shows what machine gpuq gave you and, when storage is set up, writes a file there.

    gpuq add hello-1 --repo <your ops repo> --cmd "python examples/hello_gpu.py" --no-setup
"""
import json
import os
import platform
import shutil
import subprocess
import time

print("job", os.environ.get("GPUQ_JOB_ID"), "attempt", os.environ.get("GPUQ_ATTEMPT"),
      "on", os.environ.get("GPUQ_PLATFORM"))
print("python", platform.python_version(), "on", platform.platform())
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                         capture_output=True, text=True).stdout)
else:
    print("no GPU on this machine")
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except ImportError:
    print("torch is not installed")

if os.environ.get("GPUQ_REMOTE"):
    from gpuq.storage import Remote
    remote = Remote.from_env()
    rel = f"hello/{os.environ['GPUQ_JOB_ID']}.json"
    info = remote.put_json({"job": os.environ["GPUQ_JOB_ID"], "platform": os.environ["GPUQ_PLATFORM"],
                            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, rel)
    print(f"wrote {os.environ['GPUQ_REMOTE']}/{rel} ({info['size']} bytes, md5 {info['md5']})")
else:
    print("storage.remote is not set in gpuq.toml, nothing written")
