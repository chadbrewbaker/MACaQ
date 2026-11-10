"""Portable model configuration without inherited source locations."""

from pathlib import Path


def portable_config(value):
    if isinstance(value, dict):
        return {key: portable_config(item) for key, item in value.items()
                if key not in ('_name_or_path', 'name_or_path')}
    if isinstance(value, list):
        return [portable_config(item) for item in value]
    return value


def portable_model_source(source):
    """Keep Hub references; request an explicit base when the source is local."""
    source = portable_config(source)
    model = source.get('model')
    if model is not None:
        model = str(model)
        if (source.get('local_model') or Path(model).is_absolute()
                or Path(model).exists() or model.startswith(('.', '~'))):
            source.update(model=None, local_model=True)
        else:
            source['model'] = model
    return source
