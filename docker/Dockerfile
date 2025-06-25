FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.10
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "src/main.py"] 