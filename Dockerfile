FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_AUTOINSTALL=False

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libjpeg-turbo8 \
    libpng16-16 \
    libtiff5 \
    libopenjp2-7 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /app/requirements.txt

# Fail-fast: detectar problemas en build time, no en runtime
RUN python -c "import numpy as np; print('numpy ok', np.__version__)"
RUN python -c "import torch; print('torch ok', torch.__version__, '| cuda:', torch.cuda.is_available())"
RUN python -c "from ultralytics import YOLO; print('ultralytics ok')"
RUN python -c "from huggingface_hub import snapshot_download; print('hf_hub ok')"

# Precargar modelo de patentes desde HF Hub — elimina cold start
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "\
import os; \
from huggingface_hub import snapshot_download; \
snapshot_download(\
    repo_id='morsetechlab/yolov11-license-plate-detection', \
    local_dir='/app/models/license-plate', \
    local_dir_use_symlinks=False, \
    token=os.environ.get('HF_TOKEN') or None \
)"

# Verificar que el .pt quedó en la imagen
RUN python -c "\
import glob; \
weights = glob.glob('/app/models/license-plate/**/*.pt', recursive=True); \
assert weights, 'ERROR: no se encontro .pt en /app/models/license-plate'; \
print('modelo ok:', weights[0])"

COPY . /app

CMD ["python", "-u", "/app/handler.py"]
