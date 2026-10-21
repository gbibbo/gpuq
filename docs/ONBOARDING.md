# Onboarding

This guide takes you from nothing to your first job running on a free Colab or Kaggle GPU. Plan on about
an hour and a half, most of it waiting for pages to load and emails to arrive. Do the steps in order.
Each one ends with a **Check** that tells you it worked.

If you prefer to be guided, paste the prompt in [AGENT_PROMPT.md](AGENT_PROMPT.md) into ChatGPT, Claude
or Gemini and attach this file.

**One rule for the whole guide.** Tokens and keys go only into `gpuq setup`, which runs on your own
machine and hides what you type. Never paste them into a chat, an email, an issue or a commit. If one
leaks, revoke it and make a new one.

## What you will have at the end

- A private copy of gpuq on your GitHub account. It holds the code, your settings (`gpuq.toml`), the
  queue (branch `jobs`) and the status and logs of your jobs (branch `runs`).
- A small computer that is always on (a Raspberry Pi is ideal) running the gpuq dispatcher in Docker.
  It launches your jobs on Colab and Kaggle, relaunches them when a machine is lost and tells your phone
  when they finish.
- On your laptop, the `gpuq` command to add jobs and see how they are going.

## 1. What you need

| | What | Why |
| --- | --- | --- |
| Required | GitHub account | your copy of gpuq, your code, the queue |
| Required | Google account | Colab, and Google Drive for checkpoints |
| Required for Kaggle | Kaggle account with a verified phone | about 30 GPU hours a week |
| Required | An always-on computer with Docker | runs the dispatcher (section 10) |
| Required | Python 3.11 or newer and git on your laptop | the `gpuq` command |
| Optional | Hugging Face account | faster model downloads |
| Optional | The ntfy app on your phone | notifications |

You can use Colab only or Kaggle only. Set the other one to `enabled = false` in `gpuq.toml`.

**Fair use.** Use one account of each platform, your own, and run only your own jobs. Colab and Kaggle
forbid using several accounts to get around their limits, and Colab forbids sharing an account. Read
[HOW_IT_WORKS.md](HOW_IT_WORKS.md#fair-use) before you start.

## 2. Your copy of gpuq

1. Open the gpuq repository page that was shared with you and click **Use this template** then **Create a
   new repository**. Owner is your account, name `gpuq` (any name works), visibility **Private**.
   If you cannot see the button, ask for access to the repository, or clone it and push it to a new
   private repository of yours.
2. Your research code also has to be in a GitHub repository **that you own**. GitHub's fine-grained
   tokens cannot reach repositories where you are only a collaborator. If your code lives in someone
   else's repository, fork it or copy it into one of yours.
3. Clone your copy on your laptop and create your settings:

   ```
   git clone https://github.com/<you>/gpuq.git
   cd gpuq
   cp gpuq.toml.example gpuq.toml        # on Windows: copy gpuq.toml.example gpuq.toml
   ```

4. Edit `gpuq.toml`. Set `ops_repo` to `<you>/gpuq`, `code_repo` to your research repository, and
   `username` under `[kaggle]` to your Kaggle user name (section 5). Leave `remote` as `gdrive:gpuq` if
   you will follow section 6 as written.
5. Commit and push it.

   ```
   git add gpuq.toml
   git commit -m "my settings"
   git push
   ```

6. Install the command on your laptop with `pip install -e .` from the same folder. Without installing,
   `python -m gpuq ...` from that folder does the same.

**Check.** `gpuq status` answers "the dispatcher has not reported yet". That is expected for now.

## 3. GitHub token

The dispatcher and the cloud machines use this token to read your code and to write the queue, status
and logs.

1. Go to GitHub, **Settings** (your profile picture), **Developer settings**, **Personal access tokens**,
   **Fine-grained tokens**, **Generate new token**.
2. Name `gpuq`. Expiration 90 days or more (up to 366, or no expiration if your organization allows it).
   You will renew it when it expires (section 12).
3. **Repository access**, **Only select repositories**. Pick your gpuq copy and your research repository.
4. **Repository permissions**, **Contents**, **Read and write**. Leave everything else as it is.
5. **Generate token** and keep the page open. GitHub shows the token (it starts with `github_pat_`) only
   once. You will paste it in section 10. If you lose it, generate another one.

**Check.** The token appears in the list of fine-grained tokens with access to two repositories.

## 4. Colab

Nothing to do now. During `gpuq setup` (section 10) the official Colab CLI prints a link. You open it in
any browser, sign in with your Google account, approve, and paste back the code Google shows.

Use the same Google account you use for Colab in the browser. The free tier gives a T4 GPU when one is
available. When none is, gpuq waits and tries again, and sends the job to Kaggle if the job allows it.

## 5. Kaggle

1. Create an account at kaggle.com if you do not have one.
2. **Settings**, **Phone verification**. Verify your phone. Without it Kaggle gives no GPU and no internet
   to notebooks, and gpuq cannot work there.
3. **Settings**, **API**, **Generate New Token**. Copy the token and keep it for section 10. It is shown
   only once.
4. Put your Kaggle user name (the one in your profile URL, kaggle.com/<name>) in `gpuq.toml` under
   `[kaggle]` as `username`, commit and push.

`gpuq setup` will create a **private** dataset called `gpuq-credentials` in your Kaggle account. Kaggle
does not hand notebook secrets to runs started from its command line, so this dataset is how your Kaggle
jobs receive their keys. Never make it public.

**Check.** kaggle.com/settings shows your phone as verified.

## 6. Storage for checkpoints and results

A cloud machine can disappear at any moment and its disk goes with it. Your jobs save checkpoints and
results to a storage space that outlives them, through rclone. This section sets up Google Drive. Any
other storage rclone supports works too (Dropbox, OneDrive, S3, Backblaze B2 and many more). Name the
remote what you like and set `remote` in `gpuq.toml` to `<remote name>:<folder>`.

### 6a. Install rclone on your laptop

- Windows `winget install Rclone.Rclone`
- macOS `brew install rclone`
- Linux `sudo -v ; curl https://rclone.org/install.sh | sudo bash`

**Check.** `rclone version` prints 1.75 or newer.

### 6b. Your own Google client for rclone

Since 2026 rclone requires you to create your own Google client. It takes about ten minutes, once.

1. Open console.cloud.google.com with your Google account and create a project named `gpuq-rclone`.
2. **APIs & Services**, **Library**, search **Google Drive API**, **Enable**.
3. **Google Auth Platform** (also called OAuth consent screen). Start the setup.
   - **Branding**. App name `gpuq-rclone`, your email as support and developer contact. Google asks for
     a home page and a privacy policy link before it lets you publish. Any public page you control is
     fine, for example a public GitHub gist with one sentence saying the app is for your personal use.
   - **Audience**. User type **External**. Add your own email under **Test users**.
   - **Data access**. **Add or remove scopes**, add `https://www.googleapis.com/auth/drive.file`.
   - **Clients**. **Create client**, type **Desktop app**, name `rclone`. Copy the **Client ID** and the
     **Client secret**.
   - Back in **Audience**, click **Publish app** so the status says **In production**. An app left in
     **Testing** loses its permission every 7 days and your jobs would stop saving.
4. Google will warn that the app is not verified when you sign in. That is expected for a personal app.

### 6c. The rclone remote

On your laptop (replace the two values):

```
rclone config create gdrive drive scope=drive.file client_id=<CLIENT_ID> client_secret=<CLIENT_SECRET>
```

A browser opens. Sign in, click **Advanced** and **Go to gpuq-rclone (unsafe)** if Google shows the
warning, and allow access. Then create the folder:

```
rclone mkdir gdrive:gpuq
rclone lsd gdrive:
```

With the `drive.file` scope rclone sees only what rclone itself created. That keeps the rest of your
Drive out of reach of the cloud machines. It also means that input data you want your jobs to read must
be uploaded with rclone, for example `rclone copy ./data gdrive:gpuq/data`.

**Check.** `rclone lsd gdrive:` lists `gpuq`. `rclone config show gdrive` prints a section that starts
with `[gdrive]`. You will paste that section in section 10, so leave the terminal open or run the
command again then.

## 7. Hugging Face token (optional)

With a token, model downloads from Hugging Face are faster and gated models you have accepted work.
huggingface.co, **Settings**, **Access Tokens**, **Create new token**, type **Read**. Keep it for
section 10.

## 8. Other secrets your jobs need (optional)

If your code needs other keys, for example `WANDB_API_KEY` for Weights & Biases, `gpuq setup` can store
them. Your jobs receive them as environment variables. Have them at hand for section 10.

## 9. Notifications on your phone (optional)

Install the **ntfy** app (Android or iOS). In section 10 `gpuq setup` creates a private topic for you and
shows it once. In the app tap **+** and subscribe to that topic. Anyone who knows the topic can read your
notifications, so do not share it.

## 10. The always-on computer

Any computer that stays on works. A Raspberry Pi 4 or 5 with Raspberry Pi OS 64-bit is ideal. An old
laptop or a small Linux server works too, and so does a Mac or a Windows PC with Docker Desktop (on
Windows, inside WSL2). Do not use a login node of a university cluster.

1. On that computer, open a terminal (or connect to it with `ssh`).
2. Install git and Docker. On a Raspberry Pi or Debian/Ubuntu:

   ```
   sudo apt update && sudo apt install -y git
   curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

   Log out and back in (or reboot) so the last line takes effect.
3. Clone your gpuq copy. When git asks for a password, paste the GitHub token from section 3. Git does
   not keep it.

   ```
   git clone https://github.com/<you>/gpuq.git
   cd gpuq
   ```

4. Run the installer. It builds the Docker image (a few minutes the first time), starts the dispatcher
   and runs `gpuq setup`.

   ```
   ./install.sh
   ```

5. `gpuq setup` walks through steps 1 to 8 below. Each value is checked as soon as you enter it, and
   what you type for a token does not show on the screen.
   1. GitHub token from section 3. It also makes a test commit to your branch `runs`.
   2. Colab sign-in from section 4. Open the link, sign in, paste the code.
   3. Kaggle token from section 5. It shows your weekly GPU quota.
   4. rclone section from 6c. Paste the output of `rclone config show gdrive` and end with an empty line.
   5. Hugging Face token (optional).
   6. Other secrets (optional).
   7. ntfy topic (optional). Subscribe to it in the app, a test notification arrives.
   8. The private Kaggle dataset with your credentials.

   You can run any step again later, for example `./install.sh setup --only kaggle`.

**Check.** `docker exec gpuq gpuq doctor` prints `[ok]` on every line, and on your laptop `gpuq status`
shows the dispatcher and the time of its last report.

Everything the dispatcher keeps lives in `~/gpuq-data` on that computer. `secrets/` holds your keys and
only your user can read it. The container restarts by itself after a reboot.

## 11. Your first jobs

On your laptop, from your gpuq folder.

1. A first job that shows what machine you got and writes a file to your storage.

   ```
   gpuq add hello-1 --repo <you>/gpuq --cmd "python examples/hello_gpu.py" --no-setup
   gpuq status
   ```

   Within a minute or two the job shows as `running` on Colab. It takes a few minutes in total.
   `gpuq logs hello-1` shows its output (logs are published every 10 minutes and at the end).
2. A job that survives interruptions. It saves checkpoints and continues from the last one when gpuq
   relaunches it.

   ```
   gpuq add toy-1 --repo <you>/gpuq --cmd "python examples/resumable_train.py" --no-setup --gpu CPU
   ```

3. Your own job. Before you run anything long, read [WRITING_JOBS.md](WRITING_JOBS.md). A job that
   does not save checkpoints starts over every time its machine is lost.

   ```
   gpuq add train-1 --cmd "python train.py --config configs/base.yaml"
   gpuq add eval-1 --cmd "python eval.py" --depends-on train-1 --gpu CPU
   ```

Job ids are never reused. To run the same thing again, give it a new id (`train-2`).

**Check.** `gpuq status` shows `hello-1` as `done`, and `rclone ls gdrive:gpuq/hello` lists its file.

## 12. Day to day

| Task | How |
| --- | --- |
| See the queue and the dispatcher | `gpuq status` |
| Output of a job | `gpuq logs <id>` |
| Cancel a job | `gpuq cancel <id>`. A job running on Kaggle also needs **Cancel run** on kaggle.com, the notification has the link |
| GitHub token expired | make a new one (section 3), then `./install.sh setup --only github kaggle_dataset` on the always-on computer |
| Colab asks to sign in again | `docker exec -it gpuq colab sessions` on the always-on computer |
| rclone config changed | `./install.sh setup --only storage kaggle_dataset` |
| Update gpuq | merge the new version into your copy, push, then `./install.sh update` |
| Check everything | `docker exec gpuq gpuq doctor` |
| Dispatcher log | `docker logs --tail 100 gpuq` |
| Stop it | `docker stop gpuq` (and `docker start gpuq` to resume) |

A change you push to `gpuq.toml` reaches the dispatcher by itself within a few minutes. It restarts on
the new settings when no Colab job is running.

## 13. When something fails

| Symptom | Likely cause and fix |
| --- | --- |
| `gpuq status` says the dispatcher has not reported | the container is not running or has no GitHub token. `./install.sh status`, then `docker exec gpuq gpuq doctor` |
| A job stays `pending` | its dependencies are not done, its platform is disabled or busy, or Colab gives no GPU right now. `gpuq status` shows the lanes |
| `setup_error` | the setup command or the checkout failed. `gpuq logs <id>` shows why. Often a missing package in `requirements.txt` or a wrong `ref` |
| `job_error` | your command exited with an error. `gpuq logs <id>` |
| Many `interrupted` attempts | the machine was lost each time. Make sure the job resumes (WRITING_JOBS.md). Big uploads from Colab, see HOW_IT_WORKS.md |
| Kaggle jobs fail at once | phone not verified, the credentials dataset missing, or the weekly quota spent. `docker exec gpuq gpuq doctor` |
| Notification "Colab login needed" | `docker exec -it gpuq colab sessions` and sign in again |
| Jobs stopped saving to Drive after a week | the Google app is still in Testing. Publish it (6b), then redo 6c and `./install.sh setup --only storage kaggle_dataset` |
