# Writing jobs that survive

Free cloud machines disappear. Colab reclaims VMs when it needs them, Kaggle stops sessions at 12 hours,
connections drop. gpuq notices, starts a fresh machine and runs your command again from the beginning.
Your code decides whether "again" means starting over or continuing. This page is about continuing.

## What a job is

A job is a shell command run at the root of your code repository, on a fresh machine, after gpuq has

1. checked out the branch, tag or commit you asked for (`--ref`, default from `gpuq.toml`),
2. run the setup command (`--setup`, default `pip install -r requirements.txt` when that file exists),
3. put your secrets and these variables in the environment.

| Variable | Value |
| --- | --- |
| `GPUQ_JOB_ID` | the job id, e.g. `train-1` |
| `GPUQ_ATTEMPT` | 1 for the first attempt, 2 after the first interruption, and so on |
| `GPUQ_PLATFORM` | `colab` or `kaggle` |
| `GPUQ_REMOTE` | your storage, e.g. `gdrive:gpuq` (when `storage.remote` is set) |
| `RCLONE_CONFIG` | path of the rclone config on the machine |
| `GPUQ_SCRATCH` | a local folder for temporary files, lost with the machine |
| `GPUQ_CODE_SHA` | the commit that is running |
| `HF_TOKEN`, `GITHUB_TOKEN` and your own secrets | as set in `gpuq setup` |

The `gpuq` package is importable from the job (gpuq adds it to `PYTHONPATH`), so `from gpuq.storage
import ...` works without installing anything.

Exit code 0 means done. Any other exit code counts as a failure of the job. After `max_failures`
failures (default 2) gpuq stops retrying and notifies you. An interruption (the machine was lost) does
not count as a failure. gpuq relaunches up to `max_interruptions` times (default 10).

## The three rules

1. **Everything that must survive goes to the remote.** The local disk is gone after an interruption.
2. **Resuming is the normal path.** On start, look for the last checkpoint and continue from it. The
   first attempt is the special case where there is none.
3. **Finished work is not redone.** Before a step, check whether its output is already on the remote.

A job that follows them can be cut at any moment, any number of times, and ends with the same result.

## Checkpoints with gpuq.storage

`Remote` uploads atomically. It writes `<name>.part`, renames it and checks size and MD5, so a file cut
halfway is never mistaken for a good one. `Checkpoints` keeps the last few checkpoints of a run.

```python
import argparse, os
import torch
from gpuq.storage import Remote, Checkpoints

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=os.environ.get("GPUQ_JOB_ID", "local"))  # names the checkpoints
run = ap.parse_args().run
remote = Remote.from_env()
ckpt = Checkpoints(remote, f"checkpoints/{run}", keep=2)
local = os.path.join(os.environ.get("GPUQ_SCRATCH", "."), "ckpt.pt")

model, opt = build_model(), build_optimizer()
step = 0
if ckpt.load(local) is not None:              # None on the first attempt
    state = torch.load(local, map_location="cpu")
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    step = state["step"]
    print(f"resumed at step {step}", flush=True)

for batch in batches(start=step):
    loss = train_step(model, opt, batch)
    step += 1
    if step % 500 == 0:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, local)
        ckpt.save(step, local)

torch.save(model.state_dict(), local)
remote.put(local, f"models/{run}/final.pt")
```

Name checkpoints after the run, not the job. Job ids are never reused, so a new job (`train-2
--cmd "python train.py --run base"`) can continue a run that an earlier job (`train-1`) started.

Save as often as losing the work in between would hurt. Every 10 to 30 minutes of compute is a good
start. Also save the data loader position and the random state if your results must not depend on
where the cuts happened.

## Work in pieces

For jobs that process many items (feature extraction, evaluation, inference), split the work in shards
and mark each shard done by uploading its output.

```python
from gpuq.storage import Remote
remote = Remote.from_env()
done = set(remote.ls(f"features/{run}"))
for i, shard in enumerate(shards(items, size=250)):
    name = f"shard_{i:05d}.npz"
    if name in done:
        continue
    path = compute(shard)
    remote.put(path, f"features/{run}/{name}")
```

## Logs and results

Print progress with `flush=True` (or run Python with `-u`). gpuq shows the output with `gpuq logs <id>`.
It publishes the end of the log every 10 minutes and at the end of each attempt. Secrets are replaced by
`***`, but do not print them anyway.

Small results (a JSON with metrics, a CSV) can also go to the `runs` branch of your gpuq repository, next
to the logs. Use the remote for anything large.

## Dependencies between jobs

`--depends-on a b` makes a job wait until `a` and `b` are done. If one of them fails or is cancelled,
the job is marked `blocked` and you get a notification. A typical layout is training on a GPU and
evaluation on a CPU afterwards.

```
gpuq add train-1 --cmd "python train.py"
gpuq add eval-1 --cmd "python eval.py --run train-1" --depends-on train-1 --gpu CPU
```

## Where a job runs

- `--platform auto` (default) takes Colab when it is free, because Colab does not spend Kaggle's weekly
  GPU quota, and Kaggle when Colab is busy or gives no GPU.
- `--gpu CPU` jobs go to Kaggle, unless you set `colab.cpu_jobs = true` (see HOW_IT_WORKS.md for why).
- Once a job has run on a platform, its later attempts stay there. A resumed run keeps the same kind of
  GPU, which avoids small numerical differences between, say, a Colab T4 and Kaggle's two T4s.

## Checklist before a long run

- [ ] Run it short first (a few minutes, `--steps 50` or similar) and look at `gpuq logs`.
- [ ] Interrupt it on purpose. Cancel it while it runs, add it again with a new id and the same
      `--run`, and check in `gpuq logs` that it resumed.
- [ ] Checkpoints and outputs appear on the remote (`rclone ls gdrive:gpuq/checkpoints`).
- [ ] The run fits in 12 hours per attempt, or it resumes.
