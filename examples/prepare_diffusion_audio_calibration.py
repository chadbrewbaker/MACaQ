"""Download original CC0 synthetic-sound captions from HF.

Only bounded prefixes of WebDataset shards are streamed; no audio is saved.
Original generator descriptions are preferred over model-rewritten captions.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
import tarfile
import tempfile
import urllib.request

REPO = "laion/ProgrammaticAudio"
REVISION = "bb3ab31b24c095f864cd0b55da1272f59d9f9db4"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration-count", type=int, default=64)
    p.add_argument("--evaluation-count", type=int, default=16)
    a = p.parse_args()
    if a.output.exists() or min(a.calibration_count, a.evaluation_count) < 1:
        p.error("use a fresh output and positive counts")
    subsets = dict(calibration=[], evaluation=[])
    targets = dict(calibration=a.calibration_count, evaluation=a.evaluation_count)
    seen = set()
    shards = []
    shard_order = list(range(1, 27))
    random.Random(73).shuffle(shard_order)
    for shard in shard_order:
        name = f"train/{shard:04d}.tar"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
        # Bound each transfer even when the server ignores Range: the tar reader
        # receives at most this prefix. JSON metadata follows each audio member.
        request = urllib.request.Request(url, headers={"Range": "bytes=0-16777215"})
        with urllib.request.urlopen(request, timeout=60) as response:
            import io
            prefix = io.BytesIO(response.read(16 * 1024**2))
        accepted = 0
        with tarfile.open(fileobj=prefix, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                if member.size > 1024**2:
                    raise ValueError("unexpected oversized caption metadata")
                raw = archive.extractfile(member).read()
                row = json.loads(raw)
                text = " ".join((row.get("original_text") or row.get("text", "")).split())
                key = text.casefold()
                if not text or key in seen or row.get("category") == "special":
                    continue
                split = "evaluation" if accepted % 4 == 3 else "calibration"
                accepted += 1
                seen.add(key)
                if len(subsets[split]) < targets[split]:
                    subsets[split].append(dict(prompt=text, seed=12000 + shard * 10 + accepted,
                        source=dict(dataset=REPO, revision=REVISION, license="CC0-1.0",
                                    split="train", shard=name, sample=row.get("__key__"),
                                    category=row.get("category"), method=row.get("generation_method"),
                                    metadata_sha256=hashlib.sha256(raw).hexdigest())))
                if accepted >= 4:
                    break
        shards.append(name)
        if all(len(subsets[k]) == targets[k] for k in targets):
            break
    if any(len(subsets[k]) != targets[k] for k in targets):
        raise ValueError("insufficient unique captions in the bounded shard prefixes")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.output.parent) as temp:
        pending = Path(temp) / "captions"
        pending.mkdir()
        for split, entries in subsets.items():
            (pending / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in entries))
        (pending / "manifest.json").write_text(json.dumps(dict(
            dataset=REPO, revision=REVISION, license="CC0-1.0", shards=shards,
            counts={key: len(rows) for key, rows in subsets.items()},
            categories=sorted({row["source"]["category"] for rows in subsets.values() for row in rows}),
            audio_saved=False, split_policy="Distinct sample identities and deduplicated original captions.",
            limitation="Synthetic-sound pilot; not representative of speech, music or all environmental audio.",
        ), indent=2) + "\n")
        pending.rename(a.output)
    print(a.output)


if __name__ == "__main__":
    main()
