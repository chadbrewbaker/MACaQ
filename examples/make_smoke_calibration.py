"""Write a small procedural corpus for integration tests, without external data.

All templates were written for this project. Numbers and objects are generated
locally; no dataset, quotation or other source material is downloaded. This is
a smoke-test corpus, not evidence of production calibration quality.
"""
import argparse
import random
from pathlib import Path


def make_text(seed=42, count=128):
    rng = random.Random(seed)
    objects = ("wooden box", "glass bowl", "metal cylinder", "stone sphere")
    colors = ("red", "green", "blue", "yellow")
    paragraphs = []
    for i in range(count):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        obj, color = rng.choice(objects), rng.choice(colors)
        paragraphs.append(
            f"Experiment {i + 1}. A {color} {obj} stands on a plain table. "
            f"There are {a} items in the first container and {b} in the second. "
            f"Together they contain {a + b} items. Moving an item between them "
            "does not change the total. Measure the result before drawing a conclusion.\n"
            f"Domanda: quanto fa {a} più {b}? Risposta: {a + b}. "
            "Un modello compatto deve conservare il significato delle istruzioni.\n"
            f"def add_{i}(x, y):\n    return x + y\n"
            "A sensor reports temperature once per minute. Store its measurements "
            "with a timestamp, compare repeated readings, and report missing values "
            "explicitly. A saved model should produce the same result after loading."
        )
    return "\n\n".join(paragraphs) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_text(args.seed), encoding="utf-8")
    print(args.output)
