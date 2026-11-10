"""Download an English Common Voice text subset for Chatterbox Turbo."""

import argparse
import json
from pathlib import Path
import tempfile

from prepare_audio_calibration import collect, REPO, REVISION
from qwen3_tts import sha, write_json


def prepare(output, calibration_count=32, evaluation_count=8):
    if output.exists() or min(calibration_count, evaluation_count) < 1:
        raise ValueError("choose a fresh output directory and positive counts")
    result = dict(calibration=[], evaluation=[])
    seen = set()
    sources = []
    for purpose, split, count in (("calibration", "train", calibration_count),
                                  ("evaluation", "dev", evaluation_count)):
        rows, provenance = collect("en", count, seen, split)
        sources.append(dict(purpose=purpose, **provenance))
        for i, row in enumerate(rows):
            result[purpose].append(dict(id=f"cv17-en-{split}-{row['source_row']}",
                text=row["text"], language="English", speaker="builtin", instruct="",
                seed=80000 + i, source=dict(dataset=REPO, revision=REVISION,
                    license="CC0-1.0", split=split,
                    **{k: v for k, v in row.items() if k != "text"})))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        pending = Path(tmp) / "data"
        pending.mkdir()
        write_json(pending / "prompts.json", result)
        write_json(pending / "manifest.json", dict(dataset=REPO, revision=REVISION,
            license="CC0-1.0", sources=sources, calibration_count=calibration_count,
            evaluation_count=evaluation_count, prompts_sha256=sha(pending / "prompts.json")))
        pending.rename(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration-count", type=int, default=32)
    p.add_argument("--evaluation-count", type=int, default=8)
    a = p.parse_args()
    prepare(a.output, a.calibration_count, a.evaluation_count)
