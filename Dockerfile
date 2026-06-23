# Worker GPU do RunPod — VoxCPM2 (OpenBMB, Apache 2.0).
# 30 idiomas incl PT/ES/EN, voice cloning zero-shot + "Ultimate Cloning"
# com transcript de referencia.
#
# Base: PyTorch 2.7.1 + CUDA 12.8 (Blackwell + Hopper support).
# Modelo VoxCPM2 = 2B params, ~8GB VRAM.

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV HF_HOME=/app/hf \
    MP3_BITRATE=64k \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System libs (soundfile/ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# VoxCPM + RunPod handler.
# IMPORTANTE: omnivoice-worker funciona perfeitamente apenas com
# `--upgrade-strategy=only-if-needed`. Reproduzimos o mesmo pattern aqui.
# Force-reinstall do trio torch+tv+ta (chatterbox pattern) quebrava o runpod
# lib silenciosamente — workers ficavam "idle/ready" mas nunca pegavam jobs.
# So forcamos torchvision matched (que era o causador do torchvision::nms
# crash em workers que importam transformers > LlamaModel > image_utils).
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
      runpod voxcpm soundfile numpy huggingface_hub

# Reinstala torchvision matched com torch da imagem base (2.7.1+cu128).
# --no-deps pra nao tocar nas outras libs.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
      --index-url https://download.pytorch.org/whl/cu128 \
      torchvision==0.22.1

# Pre-baixa pesos (cold start nao precisa baixar). Modo CPU pra evitar
# need de GPU no build host. Se falhar, download cai no runtime.
RUN python -c "\
import os; os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='600';\
try:\
    from huggingface_hub import snapshot_download;\
    snapshot_download(repo_id='openbmb/VoxCPM2', cache_dir='/app/hf');\
    print('PRE-BAKE OK');\
except Exception as e:\
    print(f'PRE-BAKE SKIP: {type(e).__name__}: {str(e)[:200]}')\
" || true

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
