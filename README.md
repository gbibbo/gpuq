# gpuq

A personal job queue for free Colab and Kaggle GPUs. You add jobs from your laptop. A small computer
that is always on (a Raspberry Pi is enough) launches them on Colab and Kaggle, follows them, relaunches
them when a machine is lost, and tells your phone when they finish.

```
gpuq add train-1 --cmd "python train.py --epochs 20"
gpuq add eval-1  --cmd "python eval.py" --depends-on train-1 --gpu CPU
gpuq status
gpuq logs train-1
```

It was built to run the experiments of a paper on free GPUs and is shared so that colleagues can run
their own. Each person runs their own copy, with their own accounts.

## What it does

- **Two platforms in parallel.** One job at a time on Colab (free T4) and one on Kaggle (about 30 GPU
  hours a week). A job can ask for either, or let gpuq pick.
- **Survives lost machines.** Colab VMs and Kaggle sessions end without warning. gpuq notices, starts a
  fresh machine and runs the job again. With checkpoints, your job continues where it stopped
  ([WRITING_JOBS.md](docs/WRITING_JOBS.md)).
- **No browser tabs.** It drives the official Colab CLI and the Kaggle CLI. Nothing has to stay open on
  your laptop.
- **Dependencies, retries, cancellation.** `--depends-on`, a cap on failures and on interruptions,
  `gpuq cancel`.
- **Everything in your GitHub repo.** The queue, the status of every job and the end of each attempt's
  log live on two branches of your private copy. You can follow your jobs from anywhere.
- **Secrets stay on your machine.** Tokens are typed into `gpuq setup` on the always-on computer and are
  replaced by `***` in everything published.

## Getting started

Follow [docs/ONBOARDING.md](docs/ONBOARDING.md). It takes about an hour and a half and covers the
accounts, the keys, the always-on computer and your first jobs. To be guided by an AI assistant, paste
[docs/AGENT_PROMPT.md](docs/AGENT_PROMPT.md) into ChatGPT, Claude or Gemini.

You need GitHub, Google and Kaggle accounts (Kaggle with a verified phone), an always-on computer with
Docker (Raspberry Pi, Linux box, Mac, or Windows with WSL2) and Python 3.11 or newer on your laptop.

## Documents

| | |
| --- | --- |
| [ONBOARDING.md](docs/ONBOARDING.md) | step by step, from nothing to the first job |
| [WRITING_JOBS.md](docs/WRITING_JOBS.md) | how to make a job resume after an interruption |
| [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | design, security, fair use, known issues |
| [AGENT_PROMPT.md](docs/AGENT_PROMPT.md) | prompt for an AI assistant that guides the setup |

## Fair use

Use your own accounts, one per platform, for your own work. Colab and Kaggle forbid using several
accounts to get around their limits, and Colab's free tier is meant for interactive use. Read the fair
use section of [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#fair-use) before you rely on gpuq.

## Layout

```
gpuq/            the package: CLI, dispatcher, VM-side code, storage helpers
docker/          the dispatcher's image
install.sh       installs and updates the dispatcher on the always-on computer
examples/        a first job and a resumable toy training run
tests/           run with python -m pytest (fake Colab and Kaggle CLIs, no network)
gpuq.toml        your settings (copy gpuq.toml.example)
```

## Status

Version 0.1, October 2026. The dispatcher design has run the full experiments of one paper on Colab
and Kaggle. This general version passes its tests against simulated Colab and Kaggle CLIs. Report
problems to the person who shared it with you.
