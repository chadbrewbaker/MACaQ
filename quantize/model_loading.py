"""Explicit trust and revision policy for optional custom model code."""
from pathlib import Path
import re


def loading_options(model, revision=None, trust_remote_code=False):
    if trust_remote_code and not Path(model).is_dir() and not re.fullmatch(r'[0-9a-fA-F]{40}', revision or ''):
        raise ValueError('custom remote code requires an immutable 40-character --revision')
    options = {'trust_remote_code': bool(trust_remote_code)}
    if revision:
        options['revision'] = revision
        options['code_revision'] = revision
    return options
