"""RunPod serverless worker — VoxCPM2 (OpenBMB, Apache 2.0).

30 idiomas incl PT/ES/EN. Voice cloning:
  - Zero-shot: passa apenas reference_wav_path
  - Ultimate cloning: passa prompt_wav_path + prompt_text (mais fidelidade)

Output: mp3 64k (compativel com gateway que ja espera esse formato).
Sample rate nativo: 48kHz studio quality.
"""
import base64
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import soundfile as sf
import torch
import runpod
from voxcpm import VoxCPM

DEVICE = os.environ.get("DEVICE", "cuda")
MP3_BITRATE = os.environ.get("MP3_BITRATE", "64k")
MODEL_ID = os.environ.get("MODEL_ID", "openbmb/VoxCPM2")
CFG_VALUE = float(os.environ.get("CFG_VALUE", "2.0"))
INFERENCE_TIMESTEPS = int(os.environ.get("INFERENCE_TIMESTEPS", "10"))

print(f"[boot] python={sys.version.split()[0]} torch={torch.__version__} "
      f"cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[boot] cuda_device={torch.cuda.get_device_name(0)} "
          f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

print(f"[boot] carregando VoxCPM ({MODEL_ID}, device={DEVICE}) ...", flush=True)
_t0 = time.time()
try:
    # load_denoiser=False reduz VRAM (denoiser e opcional pra cleanup)
    model = VoxCPM.from_pretrained(MODEL_ID, load_denoiser=False)
    if DEVICE == "cuda" and hasattr(model, "to"):
        model = model.to(DEVICE)
    SR = int(model.tts_model.sample_rate)
    print(f"[boot] modelo carregado em {time.time() - _t0:.1f}s | sr={SR}Hz", flush=True)
except Exception as e:
    print(f"[boot] ERRO no model load: {type(e).__name__}: {e}",
          flush=True, file=sys.stderr)
    traceback.print_exc()
    sys.stderr.flush()
    sys.stdout.flush()
    raise

_ref_cache = {}  # sha256 -> path da ref no disco


def _get_ref_path(ref_audio_b64, ref_text):
    key = hashlib.sha256((ref_audio_b64[:1024] + "|" + ref_text).encode()).hexdigest()
    cached = _ref_cache.get(key)
    if cached and os.path.exists(cached):
        return cached
    path = os.path.join(tempfile.gettempdir(), f"voxcpm_ref_{key[:16]}.wav")
    with open(path, "wb") as f:
        f.write(base64.b64decode(ref_audio_b64))
    _ref_cache[key] = path
    return path


def _handler_impl(job):
    inp = job["input"]
    texts = inp["texts"]
    ref_audio_b64 = inp["ref_audio_b64"]
    ref_text = inp.get("ref_text", "")
    # VoxCPM nao usa language_id explicito — auto-detecta do texto (como OmniVoice).
    # Aceita o param por compat com Chatterbox mas ignora.
    _language = inp.get("language") or inp.get("lang") or "en"

    cfg = float(inp.get("cfg_value") or CFG_VALUE)
    steps = int(inp.get("inference_timesteps") or INFERENCE_TIMESTEPS)

    ref_path = _get_ref_path(ref_audio_b64, ref_text)

    # Ultimate cloning quando temos ref_text (mais fidelidade no timbre).
    # Fallback pra zero-shot quando nao temos.
    gen_kwargs = {"cfg_value": cfg, "inference_timesteps": steps}
    if ref_text and ref_text.strip():
        gen_kwargs["prompt_wav_path"] = ref_path
        gen_kwargs["prompt_text"] = ref_text
        gen_kwargs["reference_wav_path"] = ref_path
    else:
        gen_kwargs["reference_wav_path"] = ref_path

    pieces = []
    durations = []
    t0 = time.time()
    for txt in texts:
        wav = model.generate(text=txt, **gen_kwargs)
        # VoxCPM ja retorna numpy array
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().detach().cpu().numpy()
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        pieces.append(arr)
        durations.append(round(len(arr) / SR, 3))
    gen_seconds = round(time.time() - t0, 3)

    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    audio_seconds = round(len(audio) / SR, 3)

    # WAV -> MP3 via ffmpeg pipe (mesmo padrao do OmniVoice/Chatterbox worker)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, SR, format="WAV", subtype="PCM_16")
    wav_buf.seek(0)
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-i", "pipe:0",
         "-c:a", "libmp3lame", "-b:a", MP3_BITRATE,
         "-f", "mp3", "pipe:1"],
        input=wav_buf.read(), capture_output=True, check=True,
    )
    mp3_bytes = proc.stdout

    return {
        "audio_b64": base64.b64encode(mp3_bytes).decode(),
        "audio_format": "mp3",
        "sample_rate": SR,
        "n_chunks": len(texts),
        "chunk_durations": durations,
        "gen_seconds": gen_seconds,
        "audio_seconds": audio_seconds,
        "rtf": round(gen_seconds / audio_seconds, 4) if audio_seconds else None,
        "model": MODEL_ID,
        "mode": "ultimate" if ref_text and ref_text.strip() else "zero-shot",
    }


def handler(job):
    """Wrapper que captura QUALQUER exception e retorna no payload em vez de crashar.
    Permite ver o erro mesmo sem acesso aos logs INFO do RunPod console."""
    try:
        return _handler_impl(job)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[handler ERROR] {type(e).__name__}: {e}\n{tb}",
              flush=True, file=sys.stderr)
        return {
            "error": True,
            "exception_type": type(e).__name__,
            "exception_message": str(e),
            "traceback": tb,
        }


runpod.serverless.start({"handler": handler})
