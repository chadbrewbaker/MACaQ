"""Calibrate dense Qwen3.5-family text blocks and export a MACaQ checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from checkpoint_config import portable_config

from text_stream import Streamer as HFCalibrationStream
from macaq_mxfp4 import mxfp4_realquant_h, OCPDenseW4A4Linear, _set_module

GATES = ('linear_attn.in_proj_a', 'linear_attn.in_proj_b', 'linear_attn.in_proj_ba')


def projections(block):
    return {name: m for name, m in block.named_modules()
            if isinstance(m, torch.nn.Linear) and not name.endswith(GATES)}


def calibrate(stream, block):
    targets = projections(block)
    hessians = {n: torch.zeros(m.in_features, m.in_features, device=stream.dev) for n,m in targets.items()}
    handles = []
    counts = {n:0 for n in targets}
    shared = {}
    def hook(name):
        def capture(module, args):
            x = args[0].detach().reshape(-1, module.in_features)
            key = (x.data_ptr(), tuple(x.shape), x.stride())
            if key not in shared:
                shared[key] = (x, x.float().T @ x.float())
            hessians[name].add_(shared[key][1])
            counts[name] += len(x)
        return capture
    for name, module in targets.items():
        handles.append(module.register_forward_pre_hook(hook(name)))
    try:
        with torch.inference_mode():
            for args, kwargs in stream.cached:
                shared.clear()
                block(*stream._args(args), **kwargs)
    finally:
        for h in handles: h.remove()
        shared.clear()
    if not all(counts.values()):
        raise ValueError('a projection received no calibration inputs')
    return targets, hessians, counts


def quantize(source, calibration, output, tokens=16384, chunk=512, flips=512,
             precision='w4a4'):
    if output.exists():
        raise ValueError('choose a fresh checkpoint directory')
    cfg = AutoConfig.from_pretrained(source, local_files_only=True)
    text = getattr(cfg, 'text_config', cfg)
    if text.model_type not in ('qwen3_5', 'qwen3_5_text') or hasattr(text, 'num_experts'):
        raise ValueError('this adapter supports dense Qwen3.5-family text models')
    stream = HFCalibrationStream(str(source), torch.device('cuda'), tokens, chunk, str(calibration))
    tensors = {}
    inventory = []
    started = time.perf_counter()
    # Keep the official container configuration and checkpoint prefixes. The
    # runtime's language-model-only option skips the visual tower.
    prefix = 'model.language_model.' if hasattr(cfg, 'text_config') else 'model.'
    with torch.inference_mode():
        for index in range(stream.n_layers):
            block = stream.load_block(index)
            targets, hessians, counts = calibrate(stream, block)
            for name, old in targets.items():
                if old.in_features % 256:
                    raise ValueError(f'unsupported projection width: {name}')
                art = mxfp4_realquant_h(old.weight.float(), hessians.pop(name), counts[name],
                    had_block=32, n_flip_iters=flips, ridge=1.0, pre_mode='balance',
                    pre_alpha=.75, do_flip=True, exp_screen=True, exp_offsets=(1,0,-1),
                    fast_flip=True, compile_flip=False)
                replacement = OCPDenseW4A4Linear(art['W_packed'],art['W_scales'],art['BH'],
                    old.bias,32,old.out_features,old.in_features,pre_scale=art.get('pre_scale'),
                    activation_precision=precision)
                _set_module(block,name,replacement)
                inventory.append(dict(name=f'{prefix}layers.{index}.{name}',rows=counts[name],
                                      in_features=old.in_features,out_features=old.out_features))
                del old,art
            del targets,hessians
            stream.forward_all(block, keep=True)
            for key,value in block.state_dict().items():
                if key.endswith('.B_inv_T'):
                    continue
                key = key.replace('.W_packed','.weight_packed').replace('.W_scales','.weight_scale')
                tensors[f'{prefix}layers.{index}.{key}'] = value.detach().cpu().contiguous().clone()
            del block
            torch.cuda.empty_cache()
            print(f'Quantized block {index+1}/{stream.n_layers}, {len(inventory)} projections',flush=True)
        # Embeddings, norms, head and small recurrent gates retain their source
        # tensor values and dtypes. Vision and auxiliary prediction weights are
        # outside this text-only export.
        for source_key,shard in stream.weight_map.items():
            if source_key.startswith(stream.prefix) or not (
                source_key in (prefix+'embed_tokens.weight',prefix+'norm.weight','lm_head.weight')):
                continue
            tensors[source_key] = stream.shards.get(shard).get_tensor(source_key).cpu().contiguous().clone()
    output.mkdir(parents=True)
    save_file(tensors, output/'model.safetensors', metadata={'format':'pt'})
    config=portable_config(cfg.to_dict())
    config['quantization_config']=dict(quant_method='mxfp4',macaq_dense_w4a4=True,
        macaq_activation_precision=precision,hadamard_block=32,macaq_unquantized_suffixes=list(GATES))
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    tok=AutoTokenizer.from_pretrained(source,local_files_only=True)
    tok.save_pretrained(output)
    # Preserve generation defaults while removing any inherited quantizer state.
    import shutil
    if (source/'generation_config.json').exists():
        shutil.copyfile(source/'generation_config.json',output/'generation_config.json')
    report=dict(source=str(source),calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        tokens=stream.T,chunk=chunk,flips=flips,precision=precision,layers=inventory,
        elapsed_seconds=time.perf_counter()-started)
    (output/'macaq_run.json').write_text(json.dumps(report,indent=2)+'\n')
    import sys
    from source_provenance import save_source_fingerprint
    save_source_fingerprint(source, output)
    stream.shards.close()
    print(f'Saved {len(inventory)} projections to {output}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=16384)
    p.add_argument('--chunk',type=int,default=512)
    p.add_argument('--flips',type=int,default=512)
    p.add_argument('--precision',choices=('w4a4','w4a16'),default='w4a4')
    a=p.parse_args()
    if min(a.tokens,a.chunk,a.flips)<1: p.error('counts must be positive')
    torch.set_num_threads(8)
    quantize(a.model,a.calibration,a.output,a.tokens,a.chunk,a.flips,a.precision)


if __name__=='__main__':
    main()
