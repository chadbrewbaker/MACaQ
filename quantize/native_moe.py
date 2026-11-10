"""Export packed text experts using verified native-runtime activation samples."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from checkpoint_config import portable_config

ROOT = Path(__file__).resolve().parents[1]
for directory in ('llamacpp', 'llamacpp/llama.cpp/gguf-py'):
    sys.path.insert(0, str(ROOT / directory))
from source_provenance import save_source_fingerprint
from mxfp4_helper import ShardCache, load_weight_index, QuantizedFusedExperts, _make_linear_view
from quantize_moe_qwen35 import quantize_experts_ocp, ShardedSafetensorsWriter
from moe_coverage import SoftCoverage


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_samples(source, calibration, recipe_path):
    recipe = json.loads(recipe_path.read_text())
    if recipe.get('hf_config_sha256') != sha256(source / 'config.json'):
        raise ValueError('native calibration belongs to a different source configuration')
    if recipe.get('calibration_sha256') != sha256(calibration):
        raise ValueError('native calibration text differs from the supplied corpus')
    for item in recipe['source_files']:
        path = source / Path(item['path']).name
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError('original weights differ from the native calibration source')
    samples = recipe_path.parent / 'calibration-samples.gguf'
    if sha256(samples) != recipe.get('samples_sha256'):
        raise ValueError('native activation samples failed integrity verification')
    return samples, recipe


@torch.inference_mode()
def quantize(source, calibration, recipe_path, output, flips=64, pool_rows=2048):
    import gguf
    if output.exists() or min(flips, pool_rows) < 1:
        raise ValueError('choose a fresh output and positive refinement settings')
    config = json.loads((source / 'config.json').read_text())
    text = config.get('text_config', config)
    if text['model_type'] not in ('qwen4_exp', 'qwen4_exp_text') or config.get('quantization_config'):
        raise ValueError('this native-sample adapter requires original Flash-Next weights')
    samples, native_recipe = verified_samples(source, calibration, recipe_path)
    reader = gguf.GGUFReader(samples)
    banks = {tensor.name: tensor.data for tensor in reader.tensors}
    aliases = {}
    for position in range(text['num_hidden_layers']):
        name = f'blk.{position}.ffn_gate_exps.weight'
        selected = name if name + '.samples' in banks else name.replace('ffn_gate_exps', 'ffn_gate_up_exps')
        if selected + '.samples' not in banks:
            raise ValueError(f'native routed samples are missing for block {position}')
        aliases[name] = selected
    coverage = SoftCoverage(source, banks, aliases, max_rows=pool_rows)
    index = load_weight_index(source)
    shards = ShardCache(source)
    prefix = coverage.prefix
    def load(name):
        return shards.get(index[name]).get_tensor(name)
    settings = dict(n_flip_iters=flips, ridge=1., flip=True, exp_screen=True,
                    fast_flip=True, exp_offsets=(1, 0, -1), pre_alpha=.75, calib_mode='router')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        pending = Path(temporary) / 'checkpoint'
        writer = ShardedSafetensorsWriter(pending, 2_000_000_000)
        for position in range(text['num_hidden_layers']):
            layer = f'{prefix}layers.{position}.mlp.'
            x = coverage.pool(str(position), (str(position), 'cuda:0'), 'cuda:0').bfloat16()
            router = load(layer + 'gate.weight').to('cuda:0', torch.bfloat16)
            probabilities = torch.softmax(torch.nn.functional.linear(x, router).float(), dim=-1)
            gate_up = load(layer + 'experts.gate_up_proj').to('cuda:0', torch.bfloat16)
            down = load(layer + 'experts.down_proj').to('cuda:0', torch.bfloat16)
            experts = QuantizedFusedExperts(text['num_experts'], torch.nn.functional.silu)
            for expert in range(text['num_experts']):
                experts.gate_up_proj.append(_make_linear_view(gate_up.shape[-1], gate_up.shape[-2], gate_up[expert]))
                experts.down_proj.append(_make_linear_view(down.shape[-1], down.shape[-2], down[expert]))
            tensors = quantize_experts_ocp(experts, x, probabilities, settings, 'cuda:0')
            for name, tensor in tensors.items():
                writer.add(f'model.layers.{position}.mlp.experts.{name}', tensor)
            del experts, tensors, gate_up, down, probabilities, x, router
            gc.collect();torch.cuda.empty_cache()
            print(f'Quantized native-calibrated block {position + 1}/{text["num_hidden_layers"]}', flush=True)
        for name in index:
            if '.mlp.experts.' in name or not (name.startswith(prefix) or name == 'lm_head.weight'):
                continue
            # Auxiliary prediction and visual weights are outside this text export.
            if '.mtp.' in name or name.startswith('mtp.'):
                continue
            saved_name = 'model.' + name[len(prefix):] if name.startswith(prefix) else name
            writer.add(saved_name, load(name))
        writer.finalize()
        saved_config = portable_config(text)
        saved_config.update(architectures=['Qwen4ExpForCausalLM'], model_type='qwen4_exp_text',
            quantization_config=dict(quant_method='mxfp4', macaq_moe=True, rotation_block_size=32))
        (pending / 'config.json').write_text(json.dumps(saved_config, indent=2) + '\n')
        for path in source.iterdir():
            if path.is_file() and path.name.startswith(('tokenizer', 'vocab', 'merges', 'special_tokens',
                                                       'added_tokens', 'chat_template', 'generation_config', 'LICENSE')):
                shutil.copyfile(path, pending / path.name)
        save_source_fingerprint(source, pending)
        recipe = dict(precision='w4a4', text_only=True, expert_layers=text['num_hidden_layers'],
            experts_per_layer=text['num_experts'], flips=flips, pool_rows=pool_rows,
            calibration_sha256=sha256(calibration), native_samples_sha256=sha256(samples),
            teacher_quantized=native_recipe['teacher_quantized'],
            method='soft-router-weighted original expert projections from native activation reservoirs')
        (pending / 'macaq_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
        pending.rename(output)
    coverage.close();shards.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'calibration', 'native-recipe', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--flips', type=int, default=64)
    p.add_argument('--pool-rows', type=int, default=2048)
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    if a.gpu < 0:p.error('GPU index must be nonnegative')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    torch.set_num_threads(4)
    quantize(a.model, a.calibration, a.native_recipe, a.output, a.flips, a.pool_rows)
