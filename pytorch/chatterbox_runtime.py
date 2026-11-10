"""Persistent-cache graph replay for batch-one Chatterbox Turbo decoding."""

from functools import wraps

import torch
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


def use_decoder_graphs(model, capacity=2048):
    """Apply the same decode optimization to BF16 and MACaQ models."""
    decoder = model.tfmr
    if hasattr(decoder, "_macaq_graph_state"):
        return decoder._macaq_graph_state
    original = decoder.forward
    state = dict(cache=StaticCache(config=decoder.config, max_cache_len=capacity))
    original_head = model.speech_head.forward if hasattr(model, "speech_head") else None
    if original_head is not None:
        @wraps(original_head)
        def head(x, *args, **kwargs):
            captured = state.get("output")
            if (captured is not None and x is captured.last_hidden_state
                    and not args and not kwargs and not torch.is_grad_enabled() and not model.training):
                return state["logits"]
            return original_head(x, *args, **kwargs)
        model.speech_head.forward = head
    from chatterbox_kernels import use_decode_attention, use_speech_activations
    use_decode_attention(decoder, state["cache"])
    use_speech_activations(decoder)

    @wraps(original)
    def forward(*args, **kwargs):
        x = kwargs.get("inputs_embeds")
        incoming = kwargs.get("past_key_values")
        if (args or model.training or torch.is_grad_enabled() or x is None or x.shape[0] != 1
                or x.dtype != torch.bfloat16 or not kwargs.get("use_cache")
                or set(kwargs) - {"inputs_embeds", "past_key_values", "use_cache"}):
            return original(*args, **kwargs)
        if incoming is None:
            output = original(**kwargs)
            if x.shape[1] + 1001 > capacity:
                return output
            state["cache"].reset()
            positions = torch.arange(x.shape[1], device=x.device)
            for i, layer in enumerate(output.past_key_values.layers):
                state["cache"].update(layer.keys, layer.values, i, {"cache_position": positions})
            state["next_position"] = x.shape[1]
            output.past_key_values = state["cache"]
            return output
        if incoming is not state["cache"]:
            return original(**kwargs)
        position = state["next_position"]
        if x.shape[1] != 1 or position >= capacity:
            raise ValueError("speech decode graph capacity exceeded; use the eager runtime")
        if "graph" not in state:
            staged = x.clone()
            pos = torch.tensor([position], device=x.device, dtype=torch.long)
            columns = torch.arange(capacity, device=x.device)
            def run():
                mask = (columns <= pos[0]).view(1, 1, 1, capacity)
                output = original(inputs_embeds=staged, past_key_values=state["cache"],
                    cache_position=pos, position_ids=pos.unsqueeze(0),
                    attention_mask=mask, use_cache=True)
                logits = original_head(output.last_hidden_state) if original_head is not None else None
                return output, logits
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, logits = run()
            # Captured kernels hold pointers into all these allocations.
            state.update(graph=graph, staged=staged, position=pos, columns=columns, output=output, logits=logits)
        state["staged"].copy_(x)
        state["position"].fill_(position)
        state["graph"].replay()
        state["next_position"] = position + 1
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=state["output"].last_hidden_state, past_key_values=state["cache"])

    decoder.forward = forward
    decoder._macaq_graph_state = state
    return state
