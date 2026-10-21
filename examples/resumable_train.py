"""A toy training run that survives interruptions. Standard library only.

It fits y = 3x + 1 with gradient descent, saves a checkpoint to the remote every 200 steps and,
when gpuq relaunches it after the VM was lost, continues from the last checkpoint instead of
starting over. The same pattern works for PyTorch (docs/WRITING_JOBS.md).

    gpuq add toy-1 --repo <your ops repo> --cmd "python examples/resumable_train.py --steps 2000" --no-setup --gpu CPU
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

from gpuq.storage import Checkpoints, LocalRemote, Remote

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--every", type=int, default=200)
ap.add_argument("--run", default=os.environ.get("GPUQ_JOB_ID", "toy"))
ap.add_argument("--sleep", type=float, default=0.01, help="seconds per step, to make the run take a while")
args = ap.parse_args()

remote = Remote.from_env() if os.environ.get("GPUQ_REMOTE") else LocalRemote("toy_remote")
ckpt = Checkpoints(remote, f"checkpoints/{args.run}", keep=2, suffix=".json")
local = Path(os.environ.get("GPUQ_SCRATCH", ".")) / "toy_ckpt.json"

state = {"step": 0, "w": 0.0, "b": 0.0}
if ckpt.load(local) is not None:
    state = json.loads(local.read_text())
    print(f"resumed from step {state['step']}", flush=True)
else:
    print("no checkpoint, starting from scratch", flush=True)

rng = random.Random(state["step"])
while state["step"] < args.steps:
    x = rng.uniform(-1, 1)
    err = state["w"] * x + state["b"] - (3 * x + 1)
    state["w"] -= 0.05 * err * x
    state["b"] -= 0.05 * err
    state["step"] += 1
    time.sleep(args.sleep)
    if state["step"] % args.every == 0 or state["step"] == args.steps:
        local.write_text(json.dumps(state))
        ckpt.save(state["step"], local)
        print(f"step {state['step']}: w={state['w']:.3f} b={state['b']:.3f}, checkpoint saved", flush=True)

remote.put_json(state, f"results/{args.run}/final.json")
print(f"done: w={state['w']:.3f} b={state['b']:.3f}", flush=True)
