"""A small PyTorch training run that resumes after an interruption, written the way gpuq expects.

It fits a two-layer network to a synthetic regression problem, saves a checkpoint to your gpuq storage
every --every steps and, when gpuq relaunches it on a new machine, continues from the last one.

In your own repo, requirements.txt sits at the root and gpuq installs it before the job. Here it is in
this folder, so the example names it in --setup:

    gpuq add ex-1 --repo <you>/gpuq --setup "pip install -q -r examples/pytorch/requirements.txt" \
        --cmd "python examples/pytorch/train.py --run ex --steps 3000"
"""
import argparse
import os
import time

import numpy as np
import torch
from gpuq.storage import Checkpoints, Remote

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=os.environ.get("GPUQ_JOB_ID", "local"), help="names the checkpoints")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--every", type=int, default=500)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)

rng = np.random.default_rng(0)
x = torch.tensor(rng.normal(size=(4096, 16)), dtype=torch.float32, device=device)
w_true = torch.tensor(rng.normal(size=(16, 1)), dtype=torch.float32, device=device)
y = torch.tanh(x @ w_true) + 0.05 * torch.randn(4096, 1, device=device, generator=torch.Generator(device).manual_seed(0))

model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

remote = Remote.from_env()
ckpt = Checkpoints(remote, f"checkpoints/{args.run}", keep=2)
local = os.path.join(os.environ.get("GPUQ_SCRATCH", "."), "ckpt.pt")
step = 0
if ckpt.load(local) is not None:
    state = torch.load(local, map_location=device)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    step = state["step"]
    print(f"resumed from step {step}", flush=True)
else:
    print("no checkpoint, starting from scratch", flush=True)

t0 = time.time()
while step < args.steps:
    idx = torch.randint(0, x.shape[0], (256,), device=device)
    loss = torch.nn.functional.mse_loss(model(x[idx]), y[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    step += 1
    if step % args.every == 0 or step == args.steps:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, local)
        ckpt.save(step, local)
        print(f"step {step} loss {loss.item():.4f}, checkpoint saved ({time.time() - t0:.0f} s)", flush=True)

with torch.no_grad():
    final = torch.nn.functional.mse_loss(model(x), y).item()
remote.put_json({"run": args.run, "steps": step, "mse": final, "device": device},
                f"results/{args.run}/metrics.json")
print(f"done: mse {final:.4f}", flush=True)
