# PyTorch: speech generation

**Contents**

- [Qwen3-TTS](#qwen3-tts)
- [Chatterbox Turbo](#chatterbox-turbo)
- [Loading and generation](#loading-and-generation)

Both adapters quantize to W4A4, save a checkpoint and generate WAV files.
Use Python 3.12 and separate environments. Run commands from the repository root.

## Qwen3-TTS

```bash
python3.12 -m venv .venv-tts
source .venv-tts/bin/activate
pip install -r requirements/tts.txt
python - <<'PYMODEL'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    revision="0c0e3051f131929182e2c023b9537f8b1c68adfe",
    local_dir="models/qwen3-tts-source",
)
PYMODEL
python pytorch/prepare_audio_calibration.py --output data/qwen3-tts
python pytorch/qwen3_tts.py quantize \
  --model models/qwen3-tts-source --prompts data/qwen3-tts/prompts.json \
  --output models/qwen3-tts-macaq --work-dir work/qwen3-tts --gpu 0
python pytorch/qwen3_tts.py generate \
  --checkpoint models/qwen3-tts-macaq --prompts data/qwen3-tts/prompts.json \
  --output outputs/qwen3-tts --gpu 0
```

The helper prepares English and Italian text from separate HF source splits.
The saved checkpoint includes the talker, code predictor, tokenizer and speech
codec. Generation uses the evaluation prompts and writes one WAV per entry.

## Chatterbox Turbo

```bash
uv venv --python 3.12 .venv-chatterbox
source .venv-chatterbox/bin/activate
uv pip install --torch-backend=cu130 \
  --overrides requirements/chatterbox-overrides.txt -r requirements/chatterbox.txt
python pytorch/chatterbox_turbo.py download --output models/chatterbox-source
python pytorch/prepare_chatterbox_calibration.py --output data/chatterbox
python pytorch/chatterbox_turbo.py quantize \
  --model models/chatterbox-source --prompts data/chatterbox/prompts.json \
  --work-dir work/chatterbox --output models/chatterbox-macaq --gpu 0
python pytorch/chatterbox_turbo.py generate \
  --checkpoint models/chatterbox-macaq --prompts data/chatterbox/prompts.json \
  --output outputs/chatterbox --gpu 0
```

Chatterbox Turbo generates English speech. The saved checkpoint includes voice
assets and the audio decoder; its watermark is retained.

## Loading and generation

Choose fresh output and work directories. Saved checkpoints load without the
original weights or a second calibration pass. Preserve the complete checkpoint
directory when moving it to another machine.

Edit the generated `prompts.json` to choose text, voice, language and seed.
Keep calibration and evaluation text separate. `--limit 1` generates one sample;
`--max-tokens` controls the output length. The default fast runtime prepares
replay graphs on first use and reuses them for subsequent requests. It processes
one utterance at a time. `--runtime eager` selects the reference execution path.

To generate with the original weights, replace `--checkpoint` with `--model`
and choose another output directory.

[Main README](../README.md)
