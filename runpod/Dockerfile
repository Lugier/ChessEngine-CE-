FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/Chess

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-venv python3-pip ca-certificates build-essential clang && \
    rm -rf /var/lib/apt/lists/*

COPY trainer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY data/requirements.txt /tmp/data-requirements.txt
RUN pip install --no-cache-dir -r /tmp/data-requirements.txt huggingface_hub

CMD ["/bin/bash"]
