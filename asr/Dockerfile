# FROM pytorch/torchserve:0.12.0-gpu
FROM huggingface/transformers-pytorch-gpu:4.35.2

WORKDIR /app
ENV HF_HOME=/app/data/huggingface
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED 1

# [for dev purposes only; align with the host user's UID/GID]
# Create a new user called 'app' and grant sudo permissions
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash app \
    && echo "app ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install dependencies
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY src /app/src

# TODO [ss] For task 2e, not sure what's the intention of removing the file,
# RUN rm /app/src/app/asr_api.py

EXPOSE 8001
CMD ["python3", "./src/run.py"]