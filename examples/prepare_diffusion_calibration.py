"""Prepare text prompts and optional conditioning images for diffusion calibration."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile

REPO = "ceyda/smithsonian_butterflies"
REVISION = "b14fd6edb25ad7646d25599565008cadc013f952"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def original_prompts():
    subjects = (
        "A small wooden boat drifts across a quiet lake",
        "A red ball rolls across a plain stone courtyard",
        "A silver train travels through a green valley",
        "A brown dog walks along a sandy beach",
        "A glass of water stands beside a moving curtain",
        "A cyclist crosses a bridge over a narrow river",
        "A yellow flower sways gently in the wind",
        "A person slowly folds a sheet of blue paper",
    )
    views = (
        "Soft morning light, steady wide shot.",
        "Warm afternoon light, slow camera pan.",
        "Overcast daylight, detailed close view.",
        "Evening light, camera gradually moves closer.",
    )
    # Assign whole subjects to separate splits, rather than merely changing seeds.
    return {
        split: [dict(prompt=f"{subject}. {view}", seed=8100 + i * 10 + j,
                     source={"type": "original", "subject_group": i})
                for i, subject in enumerate(subjects) if (i >= 6) == (split == "evaluation")
                for j, view in enumerate(views)]
        for split in ("calibration", "evaluation")
    }


def cc0_images(destination, calibration_count, evaluation_count):
    import fsspec
    import pyarrow.parquet as pq
    from PIL import Image

    total = calibration_count + evaluation_count
    entries, seen = [], set()
    # HTTP range reads retrieve only the required row groups. No Hub token or
    # legacy executable dataset loader is used.
    for shard in range(4):
        name = f"data/train-{shard:05d}-of-00004.parquet"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
        with fsspec.open(url, "rb", block_size=1024**2) as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(batch_size=16, columns=["id", "image"]):
                for row in batch.to_pylist():
                    key = str(row["id"])
                    if key in seen:
                        continue
                    seen.add(key)
                    payload = row["image"]["bytes"]
                    if not payload:
                        raise ValueError("HF subset must contain image bytes, not external links")
                    index = len(entries)
                    relative = f"images/{index:04d}.png"
                    image_path = destination / relative
                    image_path.parent.mkdir(exist_ok=True)
                    with Image.open(io.BytesIO(payload)) as image:
                        image.convert("RGB").save(image_path)
                    entries.append(dict(
                        image=relative,
                        prompt="A butterfly gently opens and closes its wings. Steady camera, natural light.",
                        seed=9100 + index,
                        source=dict(dataset=REPO, revision=REVISION, split="train",
                                    shard=name, object_id=key, license="CC0-1.0",
                                    source_image_sha256=digest(payload),
                                    image_sha256=digest(image_path.read_bytes())),
                    ))
                    if len(entries) == total:
                        return dict(calibration=entries[:calibration_count],
                                    evaluation=entries[calibration_count:])
    raise ValueError("insufficient distinct source objects")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", action="store_true", help="download the CC0 I2V pilot")
    parser.add_argument("--calibration-images", type=int, default=4)
    parser.add_argument("--evaluation-images", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or min(args.calibration_images, args.evaluation_images) < 1:
        parser.error("use a fresh output and positive subset sizes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        destination = Path(temp) / "subset"
        destination.mkdir()
        text = original_prompts()
        manifest = dict(schema=1, text_to_video=text,
                        scope="Original prompt pilot; optional narrow-domain CC0 I2V pilot. No general quality claim.")
        for split, entries in text.items():
            (destination / f"{split}.txt").write_text(
                "".join(entry["prompt"] + "\n" for entry in entries), encoding="utf-8")
        if args.images:
            manifest["image_to_video"] = cc0_images(
                destination, args.calibration_images, args.evaluation_images)
        for task in ("text_to_video", "image_to_video"):
            for split, entries in manifest.get(task, {}).items():
                (destination / f"{task}-{split}.jsonl").write_text(
                    "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        destination.rename(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
