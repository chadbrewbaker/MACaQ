"""Read local prompt/image recipes, retaining provenance and disjoint identities."""

import json
import hashlib
from pathlib import Path


def read_examples(path):
    path = Path(path)
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("every recipe entry needs a nonempty prompt")
        source = row.get("source", {})
        if not isinstance(source, dict):
            raise ValueError("recipe source must be a provenance object")
        if row.get("image"):
            image = (path.parent / row["image"]).resolve()
            if not image.is_file():
                raise ValueError(f"missing conditioning image: {image}")
            if source.get("image_sha256") and hashlib.sha256(image.read_bytes()).hexdigest() != source["image_sha256"]:
                raise ValueError("conditioning image checksum mismatch")
            row = dict(row, image=str(image))
        if "seed" in row and (not isinstance(row["seed"], int) or row["seed"] < 0):
            raise ValueError("recipe seed must be a nonnegative integer")
        examples.append(row)
    if not examples or len({bool(row.get("image")) for row in examples}) != 1:
        raise ValueError("recipe must contain consistently text-only or image-conditioned examples")
    return examples
