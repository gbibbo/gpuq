# The gpuq dispatcher: the official Google Colab CLI, the Kaggle CLI, rclone and git.
# No browser and no open ports. install.sh builds it on your always-on machine:
#   docker build -t gpuq docker/
FROM debian:trixie-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv git curl ca-certificates tini procps unzip \
    && rm -rf /var/lib/apt/lists/*
ARG RCLONE_VERSION=1.75.2
RUN arch="$(dpkg --print-architecture)" && case "$arch" in arm64|amd64) ;; armhf) arch=arm-v7 ;; esac \
    && curl -fsSL -o /tmp/rclone.zip \
       "https://github.com/rclone/rclone/releases/download/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-${arch}.zip" \
    && unzip -q -j /tmp/rclone.zip '*/rclone' -d /usr/local/bin && chmod 755 /usr/local/bin/rclone && rm /tmp/rclone.zip
# google-colab-cli 0.7.4 and kaggle 2.2.4 are the versions gpuq was tested with
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir google-colab-cli==0.7.4 kaggle==2.2.4
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 GPUQ_SECRETS=/secrets GPUQ_STATE_DIR=/data/state
COPY bootstrap.py /opt/gpuq/bootstrap.py
COPY gpuq-cli /usr/local/bin/gpuq
RUN chmod 755 /usr/local/bin/gpuq
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "/opt/gpuq/bootstrap.py"]
