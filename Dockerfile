ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

# Update and install necessary packages
RUN apt-get update -y && apt-get install -y curl wget build-essential pkg-config && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Conda Environment
ENV MINICONDA_VERSION=py312_24.9.2-0
ENV PATH=/opt/miniconda/bin:$PATH
ENV CONDA_PACKAGE=24.9.2
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda install conda=${CONDA_PACKAGE} -y && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

ARG CREATE_VLLM=false
RUN if [ "$CREATE_VLLM" = "true" ]; then \
      apt-get update -y && apt-get install -y gcc-12 g++-12 && \
      update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
      apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
      conda create -n vllm python=3.12 -y && \
      conda run -n vllm pip install --no-cache-dir vllm bitsandbytes && \
      conda clean -ay && \
      rm -rf /opt/miniconda/pkgs && \
      find / -type d -name __pycache__ | xargs rm -rf; \
    fi

ARG CREATE_OLLAMA=false
RUN if [ "$CREATE_OLLAMA" = "true" ]; then \
      apt-get update -y && apt-get install -y lshw && \
      apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
      curl -fsSL https://ollama.com/install.sh | sh; \
    fi

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY ./lmbench /app/lmbench

ARG TORCH_CPU=false
RUN if [ "$TORCH_CPU" = "true" ]; then \
      uv export --frozen --no-dev --no-editable --no-hashes \
        | grep -v -E "^(nvidia-|triton==)" \
        | sed 's/^torch==\([0-9.]*\)$/torch==\1+cpu/' \
        > /tmp/requirements.txt && \
      uv venv && \
      uv pip install -r /tmp/requirements.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match; \
    else \
      uv sync --frozen --no-dev --no-editable; \
    fi
ENV PATH="/app/.venv/bin:$PATH"
