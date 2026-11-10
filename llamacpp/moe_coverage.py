"""Router-weighted calibration for experts missed by native hard routing."""

import json
from pathlib import Path
import re
from threading import RLock

import numpy as np
import torch


class SoftCoverage:
    def __init__(self, source, banks, aliases, max_rows=128):
        from mxfp4_helper import ShardCache, load_weight_index
        self.source = Path(source)
        config = json.loads((self.source / "config.json").read_text())
        kind = config.get("text_config", config).get("model_type")
        if kind not in ("qwen3_moe", "olmoe", "qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text",
                        "qwen4_exp", "qwen4_exp_text"):
            raise ValueError(f"no router-weighted fallback adapter for {kind}")
        self.index = load_weight_index(self.source)
        self.prefix = "model.language_model." if "model.language_model.embed_tokens.weight" in self.index else "model."
        self.shards = ShardCache(self.source)
        self.banks, self.aliases = banks, aliases
        self.pools = {}
        self.max_rows = max_rows
        if max_rows < 1:
            raise ValueError('router input pool must contain at least one row')
        self.lock = RLock()

    def weight(self, name, device):
        with self.lock:
            tensor = self.shards.get(self.index[name]).get_tensor(name)
        return tensor.to(device, torch.float32)

    def inputs(self, name, expert, device):
        layer, projection = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name).groups()
        pool_key = (layer, str(device))
        with self.lock:
            x = self.pool(layer, pool_key, device)
        prefix = f"{getattr(self, 'prefix', 'model.')}layers.{layer}.mlp."
        probabilities = torch.softmax(x @ self.weight(prefix + "gate.weight", device).T, dim=-1)[:, expert]
        total = probabilities.sum()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError("the missed expert has no finite soft routing mass; extend calibration")
        if projection == "down":
            gate_name = prefix + f"experts.{expert}.gate_proj.weight"
            if gate_name in getattr(self, 'index', {gate_name: None}):
                gate_weight = self.weight(gate_name, device)
                up_weight = self.weight(prefix + f"experts.{expert}.up_proj.weight", device)
            else:
                key = prefix + "experts.gate_up_proj"
                with self.lock:
                    tensor = self.shards.get(self.index[key]).get_tensor(key)[expert]
                gate_weight, up_weight = tensor.to(device, torch.float32).chunk(2, dim=0)
            gate = x @ gate_weight.T
            up = x @ up_weight.T
            x = torch.nn.functional.silu(gate) * up
        weighted = x * (probabilities * len(x) / total).sqrt()[:, None]
        return weighted, weighted.square().sum(0)

    def pool(self, layer, pool_key, device):
        if pool_key not in self.pools:
            gate = self.aliases[f"blk.{layer}.ffn_gate_exps.weight"]
            samples = self.banks[gate + ".samples"]
            counts = self.banks[gate + ".sample_seen"]
            parts = [samples[e, :min(int(count), samples.shape[1])]
                     for e, count in enumerate(counts) if count > 0]
            if not parts:
                raise ValueError("native calibration did not capture any input for this layer")
            pool = np.concatenate(parts)
            rows = min(len(pool), getattr(self, 'max_rows', 128))
            selected = np.linspace(0, len(pool) - 1, rows, dtype=np.int64)
            self.pools[pool_key] = torch.from_numpy(pool[selected].copy()).to(device)
        return self.pools[pool_key]

    def close(self):
        self.shards.close()
