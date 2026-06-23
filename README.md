# voxcpm-worker

RunPod serverless worker rodando **VoxCPM2** (OpenBMB, Apache 2.0) pra TTS multilingual com voice cloning.

## Specs
- **Modelo:** [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — 2B params, 30 idiomas
- **License:** Apache 2.0 (livre comercial)
- **VRAM:** ~8GB
- **Sample rate nativo:** 48kHz
- **RTF (RTX 4090):** ~0.3 standard / ~0.13 com Nano-vLLM
- **Output worker:** mp3 64kbps (transcodificado via ffmpeg)

## Input do job

```json
{
  "input": {
    "texts": ["frase 1", "frase 2"],
    "ref_audio_b64": "<base64 do mp3/wav de referencia>",
    "ref_text": "(opcional) transcricao do sample p/ ultimate cloning",
    "language": "(opcional) ignorado — VoxCPM auto-detecta",
    "cfg_value": 2.0,
    "inference_timesteps": 10
  }
}
```

## Output

```json
{
  "audio_b64": "...",
  "audio_format": "mp3",
  "sample_rate": 48000,
  "n_chunks": 2,
  "chunk_durations": [3.4, 2.1],
  "gen_seconds": 1.7,
  "audio_seconds": 5.5,
  "rtf": 0.309,
  "model": "openbmb/VoxCPM2",
  "mode": "ultimate"
}
```

Em caso de erro, retorna `{"error": true, "exception_type": ..., "traceback": ...}` em vez de crashar o worker.

## Build local

```bash
docker build -t voxcpm-worker .
```

## RunPod deploy

Endpoint Queue mode, GPU 24GB+, max_workers 5, idle_timeout 5s, FlashBoot ON, container disk 8GB.

Aponta pro repo via "GitHub Repo" no setup do endpoint.
