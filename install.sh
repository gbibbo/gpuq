#!/usr/bin/env bash
# Installs the gpuq dispatcher on this machine (Raspberry Pi, Linux box, Mac, or WSL2 on Windows).
# Run it from your copy of the repo:
#
#   ./install.sh            build the image, start the dispatcher and run gpuq setup
#   ./install.sh setup      run gpuq setup again (all steps, or e.g. ./install.sh setup --only kaggle)
#   ./install.sh update     pull this clone (after gpuq upgrade on your laptop), rebuild, restart
#   ./install.sh status     container state and the last lines of its log
#   ./install.sh uninstall  stop and remove the container (keeps ~/gpuq-data with your secrets)
#
# Everything the dispatcher keeps lives in ~/gpuq-data (override with GPUQ_DATA):
#   secrets/       your keys, readable only by you
#   colab-config/  the Colab CLI's Google sign-in
#   state/         the dispatcher's state and attempt logs
set -euo pipefail

cd "$(dirname "$0")"
DATA="${GPUQ_DATA:-$HOME/gpuq-data}"
NAME=gpuq
IMAGE=gpuq

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

ops_repo() {
  # owner/name of this clone's origin on GitHub
  local url
  url="$(git remote get-url origin 2>/dev/null || true)"
  [ -n "$url" ] || die "this folder is not a clone of your gpuq repo (no git origin)"
  url="${url%.git}"
  url="${url#git@github.com:}"
  url="${url#https://github.com/}"
  url="${url#https://*@github.com/}"
  printf '%s' "$url"
}

need_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    die "Docker is not installed. On a Raspberry Pi or Debian/Ubuntu run:
  curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
  sudo usermod -aG docker \$USER     # then log out and back in
On a Mac or Windows install Docker Desktop (on Windows run this script inside WSL2)."
  fi
  docker info >/dev/null 2>&1 || die "Docker is installed but this user cannot use it. Run:
  sudo usermod -aG docker \$USER
then log out and back in (or reboot) and run ./install.sh again."
}

check_settings() {
  [ -f gpuq.toml ] || die "gpuq.toml is missing. Copy gpuq.toml.example to gpuq.toml, fill it in, commit and push"
  grep -q "^ops_repo *= *\"$(ops_repo)\"" gpuq.toml || \
    echo "warning: github.ops_repo in gpuq.toml is not $(ops_repo), the origin of this clone"
  git fetch -q origin 2>/dev/null || true
  if [ -n "$(git status --porcelain gpuq.toml 2>/dev/null)" ] || \
     [ "$(git rev-parse HEAD 2>/dev/null)" != "$(git rev-parse '@{u}' 2>/dev/null || git rev-parse HEAD)" ]; then
    echo "warning: this clone differs from GitHub. The dispatcher runs what is on GitHub, so commit and push"
    echo "         your changes to gpuq.toml (or git pull) before relying on them."
  fi
}

build() {
  say "Building the image (a few minutes the first time)"
  docker build -q -t "$IMAGE" docker/ >/dev/null
}

start() {
  mkdir -p "$DATA/secrets" "$DATA/colab-config" "$DATA/state"
  chmod 700 "$DATA/secrets" "$DATA/colab-config"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  say "Starting the dispatcher container ($NAME), restarted automatically after a reboot"
  docker run -d --name "$NAME" --restart unless-stopped \
    -e GPUQ_OPS_REPO="$(ops_repo)" \
    -v "$DATA/secrets:/secrets" \
    -v "$DATA/colab-config:/root/.config/colab-cli" \
    -v "$DATA/state:/data/state" \
    -v "$PWD:/app:ro" \
    "$IMAGE" >/dev/null
}

own() {
  # The container writes as root. Give the files back to you, still readable only by you.
  docker exec "$NAME" chown -R "$(id -u):$(id -g)" /secrets /root/.config/colab-cli 2>/dev/null || true
}

pull() {
  # git pull of this private clone with the token saved by gpuq setup, passed in a header (never in a URL)
  local token basic
  token="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("GITHUB_TOKEN", ""))' \
           "$DATA/secrets/credentials.json" 2>/dev/null || true)"
  if [ -z "$token" ]; then
    git pull --ff-only
    return
  fi
  basic="$(printf 'x-access-token:%s' "$token" | base64 | tr -d '\n')"
  GIT_TERMINAL_PROMPT=0 GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.https://github.com/.extraheader \
    GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $basic" git pull --ff-only
}

setup() {
  say "Accounts and keys, step by step (docs/ONBOARDING.md explains each one)"
  docker exec -it "$NAME" gpuq setup "$@"
  own
  docker restart "$NAME" >/dev/null
  say "Done. Follow the dispatcher with: docker logs -f $NAME"
}

case "${1:-install}" in
  install)
    need_docker; check_settings; build; start; setup ;;
  setup)
    shift; need_docker; setup "$@" ;;
  update)
    need_docker
    own
    pull || echo "warning: could not fast-forward this clone, building what is here"
    build; start; own
    say "Updated. The dispatcher restarted. Logs: docker logs -f $NAME" ;;
  status)
    need_docker
    docker ps -a --filter "name=^${NAME}$" --format '{{.Names}}  {{.Status}}'
    docker logs --tail 30 "$NAME" 2>&1 || true ;;
  uninstall)
    need_docker
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    say "Removed the container. Your secrets and state are still in $DATA (delete that folder to remove them)" ;;
  *)
    die "unknown command $1. Use install, setup, update, status or uninstall" ;;
esac
