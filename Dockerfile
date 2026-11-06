# Reproducible environment for hbf_eval (CPU only; the platform is simulated, no GPU is needed).
# Versions match the environment the reference results were produced in (Ubuntu 24.04, Python 3.13, torch 2.15.0, numpy 2.5.3, gcc 13).
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv gcc libc6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.txt .
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
ENV PATH=/opt/venv/bin:$PATH
COPY . .
CMD ["python3", "-m", "hbf_eval.bench", "list"]
