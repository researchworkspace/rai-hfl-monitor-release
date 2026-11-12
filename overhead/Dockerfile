# Emulated edge / cloud node for the overhead study (RQ4), with the library versions of pyproject.toml.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2 \
    && pip install --no-cache-dir numpy==2.4.6 scipy==1.17.1 pandas==3.1.1 requests==2.34.2 opacus==1.6.0 captum==0.9.0
ARG OPA_VERSION=v1.22.2
ARG TARGETARCH
RUN curl -fsSL -o /usr/local/bin/opa https://openpolicyagent.org/downloads/${OPA_VERSION}/opa_linux_${TARGETARCH}_static \
    && chmod +x /usr/local/bin/opa && opa version | head -1
WORKDIR /app
COPY src /app/src
COPY policies /app/policies
COPY scripts /app/scripts
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ENTRYPOINT ["python", "-I", "/app/scripts/raimon_cli.py"]
