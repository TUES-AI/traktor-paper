FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg \
        build-essential git wget curl ca-certificates vim tmux pkg-config \
        ripgrep \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev \
        iproute2 procps \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV VENV_DIR=/opt/venv
RUN python3.13 -m venv "$VENV_DIR" && \
    "$VENV_DIR/bin/python" -m pip install --upgrade pip && \
    "$VENV_DIR/bin/python" -m pip install --upgrade \
      --index-url "$PYTORCH_INDEX_URL" \
      torch torchvision
ENV PATH="$VENV_DIR/bin:$PATH" VIRTUAL_ENV="$VENV_DIR"

RUN printf '%s\n' \
      'VIRTUAL_ENV=/opt/venv' \
      'PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
      >> /etc/environment
RUN printf '%s\n' 'source /opt/venv/bin/activate || true' > /etc/profile.d/activate-venv.sh
RUN printf '%s\n' 'set -o vi' > /etc/profile.d/vi-mode.sh
RUN printf '%s\n' 'set -g mode-keys vi' > /etc/tmux.conf

RUN curl -fsSL https://tailscale.com/install.sh | sh

ENV TS_AUTHKEY="" \
    TS_HOSTNAME="gpu-box" \
    TS_TAGS="tag:gpu" \
    TS_STATE_DIR="/var/lib/tailscale" \
    TS_ACCEPT_DNS="false" \
    TS_ENABLE_SSH="true" \
    TS_EXTRA_ARGS="" \
    REPO_DIR="/proj/traktor-paper" \
    REPO_URL="https://github.com/tues-ai/traktor-paper.git" \
    GIT_CLONE_DEPTH="1"

RUN mkdir -p /var/lib/tailscale

COPY --chown=root:root <<'BASH' /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/proj
REPO_DIR="${REPO_DIR:-${APP_ROOT}/traktor-paper}"
REPO_URL="${REPO_URL:-https://github.com/tues-ai/traktor-paper.git}"
GIT_CLONE_DEPTH="${GIT_CLONE_DEPTH:-1}"

if ! git config --system --get-all safe.directory 2>/dev/null | grep -Fxq "$REPO_DIR"; then
  git config --system --add safe.directory "$REPO_DIR" 2>/dev/null || true
fi

mkdir -p "$APP_ROOT"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  rm -rf "$REPO_DIR" 2>/dev/null || true
  git clone --depth "$GIT_CLONE_DEPTH" "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch --all --prune --depth "$GIT_CLONE_DEPTH"
  git -C "$REPO_DIR" pull --rebase --autostash --depth "$GIT_CLONE_DEPTH"
fi

mkdir -p "$TS_STATE_DIR" /var/run/tailscale

nohup /usr/sbin/tailscaled \
  --tun=userspace-networking \
  --state="${TS_STATE_DIR}/tailscaled.state" \
  --socket=/var/run/tailscale/tailscaled.sock \
  >/tmp/tailscaled.log 2>&1 &

for _ in {1..100}; do
  [[ -S /var/run/tailscale/tailscaled.sock ]] && break
  sleep 0.1
done

hn="${TS_HOSTNAME:-gpu-box}"
UP_ARGS=(--hostname="$hn" --accept-dns="$TS_ACCEPT_DNS" --advertise-tags="$TS_TAGS")

if [[ "${TS_ENABLE_SSH:-true}" == "true" ]]; then
  UP_ARGS+=(--ssh)
fi

if [[ -n "${TS_AUTHKEY:-}" ]]; then
  UP_ARGS+=(--authkey="$TS_AUTHKEY")
fi

EXTRA=()
if [[ -n "${TS_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA <<<"${TS_EXTRA_ARGS}"
fi

for _ in {1..60}; do
  if tailscale up "${UP_ARGS[@]}" "${EXTRA[@]}"; then
    break
  fi
  sleep 2
done

tailscale status || true

if [[ $# -gt 0 ]]; then
  exec "$@"
else
  exec sleep infinity
fi
BASH

RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /proj
RUN mkdir -p /proj

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
