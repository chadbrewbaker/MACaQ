"""Compile Triton preprocessing into standalone native-runtime GPU kernels."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--groups', type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument('--q8-groups', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--pdl', action='store_true', help='enable dependent-launch overlap on supported devices')
    parser.add_argument('--moe', action='store_true', help='also export grouped MoE preprocessing')
    args = parser.parse_args()
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from macaq_triton_kernels import macaq_pre, macaq_moe_pre
    if triton.__version__ != '3.6.0':
        parser.error('the native launch ABI is validated with Triton 3.6.0; use the pinned PyTorch environment')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new or empty directory')
    major, minor = torch.cuda.get_device_capability(args.gpu)
    arch = major * 10 + minor
    target = GPUTarget('cuda', arch, 32)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.pdl and arch < 90:
        parser.error('dependent launch requires compute capability >= 9.0')
    manifest = dict(abi=2, arch=arch, groups=args.groups, q8_groups=args.q8_groups,
                    pdl=args.pdl, routed=args.moe, triton=triton.__version__,
                    source_sha256=hashlib.sha256(Path(__file__).with_name('macaq_triton_kernels.py').read_bytes()).hexdigest(),
                    kernels=[])
    for routed in ((False, True) if args.moe else (False,)):
        block = 64
        for quantized in (False, True):
            mode = 'q8' if quantized else 'f32'
            stem = f'h{block}-' + ('moe-' if routed else '') + mode
            groups = args.q8_groups if quantized else args.groups
            signature = {'x': '*fp32', 'prescale': '*fp32'}
            if routed:
                signature['ids'] = '*i32'
            signature.update(y='*fp32', packed='*i8', width='i32')
            if routed:
                signature.update(tokens='i32', routes='i32', input_routes='i32',
                                 padded='i32', id_stride='i32', experts='i32')
            else:
                signature.update(rows='i32', padded='i32')
            constants = dict(GROUPS=groups, Q8=quantized, PDL=args.pdl)
            if not routed:
                constants['H'] = block
            source = ASTSource(macaq_moe_pre if routed else macaq_pre,
                               signature=signature, constexprs=constants)
            kernel = triton.compile(source, target=target,
                                    options={'num_warps': groups, 'enable_fp_fusion': False, 'launch_pdl': args.pdl})
            cubin = kernel.asm['cubin']
            (args.output / (stem + '.cubin')).write_bytes(cubin)
            metadata = kernel.metadata
            if metadata.global_scratch_size or metadata.profile_scratch_size:
                raise ValueError('Native preprocessing ABI requires zero scratch allocation')
            parameters = re.search(r'\.entry\s+\w+\((.*?)\)', kernel.asm['ptx'], re.S)
            if parameters is None or parameters[1].count('.param') != (14 if routed else 9):
                raise ValueError('Unsupported Triton native-launch ABI')
            # Host-loader ABI: version, target, warps, shared bytes, groups, PDL, entry.
            (args.output / (stem + '.info')).write_text(
                f'{3 if routed else 2} {arch} {metadata.num_warps} {metadata.shared} {groups} {int(args.pdl)} {metadata.name}\n')
            manifest['kernels'].append(dict(stem=stem, name=metadata.name, abi=3 if routed else 2,
                                           shared=metadata.shared, num_warps=metadata.num_warps,
                                           sha256=hashlib.sha256(cubin).hexdigest()))
            print(f'Compiled {stem} for {arch}', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
