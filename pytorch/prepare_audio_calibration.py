"""Prepare CC0 Common Voice text for Qwen3-TTS; no audio downloads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unicodedata
import urllib.request

REPO = 'fsicoli/common_voice_17_0'
REVISION = '8262c16bf297c87a9cd88c51997c4758ed7a8ba2'
BASE = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}'


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def collect(language, count, seen, source_split):
    url = f'{BASE}/transcript/{language}/{source_split}.tsv'
    digest = hashlib.sha256()
    consumed = 0
    rows = []
    # Stop after the needed prefix, rather than downloading the entire TSV.
    with urllib.request.urlopen(url, timeout=60) as response:
        header = response.readline()
        digest.update(header); consumed += len(header)
        columns = next(csv.reader([header.decode('utf-8-sig')], delimiter='\t'))
        if 'sentence' not in columns:
            raise ValueError('source TSV has no sentence column')
        for index, raw in enumerate(response):
            digest.update(raw); consumed += len(raw)
            if consumed > 8 * 1024**2:
                raise ValueError('text prefix budget exceeded')
            values = next(csv.reader([raw.decode('utf-8')], delimiter='\t'))
            row = dict(zip(columns, values))
            text = ' '.join(row.get('sentence', '').split())
            key = normalized(text)
            if key in seen or not 30 <= len(text) <= 240 or not 6 <= len(text.split()) <= 45:
                continue
            if not any(c.isalpha() for c in text):
                continue
            seen.add(key)
            rows.append(dict(text=text, source_row=index, sentence_id=row.get('sentence_id'),
                             text_sha256=hashlib.sha256(text.encode()).hexdigest()))
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f'insufficient usable {language} sentences')
    return rows, dict(url=url, source_split=source_split, prefix_bytes=consumed,
                      consumed_prefix_sha256=digest.hexdigest(),
                      hash_scope='Only the consumed TSV prefix, not the entire source file')


def prepare(output, calibration_split="train", evaluation_split="dev"):
    if calibration_split not in ("train", "dev", "test") or evaluation_split not in ("train", "dev", "test"):
        raise ValueError("source splits must be train, dev or test")
    if calibration_split == evaluation_split:
        raise ValueError("calibration and evaluation must use different source splits")
    if output.exists():
        raise ValueError('choose a fresh output directory')
    result = dict(calibration=[], evaluation=[])
    provenance = []
    seen = set()
    for language, label, budget in [('en', 'English', 48), ('it', 'Italian', 16)]:
        for split, source_split, count in [('calibration', calibration_split, budget),
                                           ('evaluation', evaluation_split, 8)]:
            rows, source = collect(language, count, seen, source_split)
            provenance.append(dict(language=language, purpose=split, **source))
            for index, row in enumerate(rows):
                result[split].append(dict(id=f'cv17-{language}-{source_split}-{row["source_row"]}',
                    language=label, speaker='Aiden' if index % 2 == 0 else 'Ryan',
                    instruct='Speak naturally and clearly.', seed=170000 + len(result[split]),
                    source=dict(dataset=REPO, revision=REVISION, license='CC0-1.0',
                                language=language, split=source_split,
                                **{k:v for k,v in row.items() if k!='text'}),
                    text=row['text']))
    calibration = {normalized(x['text']) for x in result['calibration']}
    evaluation = {normalized(x['text']) for x in result['evaluation']}
    assert calibration.isdisjoint(evaluation)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        pending = Path(directory) / 'subset'; pending.mkdir()
        payload = (json.dumps(result, indent=2, ensure_ascii=False)+'\n').encode()
        (pending/'prompts.json').write_bytes(payload)
        manifest = dict(dataset=REPO, revision=REVISION, license='CC0-1.0',
            card_url=f'https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md',
            calibration_count=64, evaluation_count=16, audio_downloaded=False,
            calibration_split=calibration_split, evaluation_split=evaluation_split,
            selection='First eligible unique sentences in source order from separate source splits.',
            limitations='Small sentence subset, no speaker/book disjointness claim and no general quality preset.',
            prompts_sha256=hashlib.sha256(payload).hexdigest(), sources=provenance)
        (pending/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        pending.rename(output)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-split', choices=('train', 'dev', 'test'), default='train')
    parser.add_argument('--evaluation-split', choices=('train', 'dev', 'test'), default='dev')
    args = parser.parse_args()
    if args.calibration_split == args.evaluation_split:
        parser.error('calibration and evaluation must use different source splits')
    print(json.dumps(prepare(args.output, args.calibration_split, args.evaluation_split), indent=2))
