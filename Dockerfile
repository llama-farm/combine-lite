# Production Dockerfile for Fly.io deployment with CUDA support
# For local development on ARM64, run: python app.py --dev

FROM ubuntu:22.04 as base

RUN apt-get update -q && apt-get install -y ca-certificates wget curl python3 python3-pip && \
    wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    apt-get update -q && \
    apt-get install -y --no-install-recommends cuda-nvcc-12-4 libcublas-12-4 libcudnn9-cuda-12 && \
    rm -rf /var/lib/apt/lists/*


ENV PATH=/usr/local/cuda-12.4/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}

FROM base as runtime

WORKDIR /app
COPY . /app
RUN pip3 install --no-cache-dir -r requirements.txt && \
    python3 -c "import torch; import transformers; import peft; import datasets; import huggingface_hub; import openai; import boto3; import gradio; print('Preloaded all requirements successfully')"

CMD ["python3", "app.py"]