"""Batch-one CustomVoice graph replay with persistent inputs and KV storage."""

from functools import wraps

import torch
from transformers.cache_utils import DynamicCache


def use_predictor_graphs(model):
    predictor = model.talker.code_predictor
    original = predictor.forward
    entries = {}

    @wraps(original)
    def forward(*args, **kwargs):
        if args:
            return original(*args, **kwargs)
        embeds = kwargs.get('inputs_embeds')
        step = embeds.shape[1] - 2 if embeds is not None and embeds.shape[1] > 1 else kwargs.get('generation_steps')
        batch = embeds if embeds is not None else kwargs.get('input_ids')
        if predictor.training or torch.is_grad_enabled() or step is None or batch is None or batch.shape[0] != 1:
            return original(**kwargs)
        incoming = kwargs.get('past_key_values')
        legacy = incoming.to_legacy_cache() if incoming is not None else None
        if legacy and all(k is None for k, v in legacy):
            legacy = None
        if step not in entries:
            tensors = {k: v.clone() for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
            fixed = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
            fixed.pop('past_key_values', None)
            fixed['attention_mask'] = {'full_attention': None}
            tensors.pop('attention_mask', None)

            def run():
                call = dict(fixed, **tensors)
                call['past_key_values'] = DynamicCache.from_legacy_cache(legacy) if legacy else None
                return original(**call)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
            entries[step] = graph, tensors, output, legacy
        graph, tensors, output, expected_cache = entries[step]
        if legacy is not None:
            if expected_cache is None or len(legacy) != len(expected_cache):
                raise ValueError('unexpected code predictor graph cache')
            if any(a.data_ptr() != b.data_ptr() for pairs in zip(legacy, expected_cache) for a, b in zip(*pairs)):
                raise ValueError('code predictor graph cache storage changed')
        for key, target in tensors.items():
            value = kwargs[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('code predictor graph input changed')
            target.copy_(value)
        graph.replay()
        return output

    predictor.forward = forward
    return entries


def use_talker_graphs(model, capacity=512):
    from transformers.cache_utils import StaticCache
    from transformers.modeling_outputs import BaseModelOutputWithPast
    class SpeechStaticCache(StaticCache):
        # Qwen3-TTS expects a 2D mask at the outer talker boundary. The decoder
        # supplies its own explicit 4D mask after that boundary.
        @property
        def is_compileable(self):
            return False

    talker = model.talker
    decoder = talker.model
    original_forward = decoder.forward
    original_generate = talker.generate
    state = {}

    @wraps(original_generate)
    def generate(*args, **kwargs):
        embeds = kwargs.get('inputs_embeds')
        mask = kwargs.get('attention_mask')
        if (args or talker.training or torch.is_grad_enabled() or embeds is None
                or embeds.shape[0] != 1 or (mask is not None and not bool(mask.all()))):
            return original_generate(*args, **kwargs)
        needed = embeds.shape[1] + kwargs.get('max_new_tokens', capacity)
        if needed > capacity:
            return original_generate(**kwargs)
        if 'cache' not in state:
            state['cache'] = SpeechStaticCache(config=talker.config, max_cache_len=capacity)
        state['cache'].reset()
        kwargs['past_key_values'] = state['cache']
        kwargs['cache_implementation'] = None
        kwargs['disable_compile'] = True
        return original_generate(**kwargs)

    @wraps(original_forward)
    def forward(*args, **kwargs):
        x = kwargs.get('inputs_embeds')
        cache = kwargs.get('past_key_values')
        if args or x is None or cache is not state.get('cache'):
            return original_forward(*args, **kwargs)
        if x.shape[:2] != (1, 1):
            columns = torch.arange(capacity, device=x.device)
            kwargs['attention_mask'] = (columns[None, :] <= kwargs['cache_position'][:, None]).view(1, 1, x.shape[1], capacity)
            return original_forward(**kwargs)
        if 'graph' not in state:
            tensors = {k: v.clone() for k, v in kwargs.items()
                       if isinstance(v, torch.Tensor) and k != 'attention_mask'}
            fixed = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
            fixed.pop('attention_mask', None)
            columns = torch.arange(capacity, device=x.device)
            def run():
                mask = (columns <= tensors['cache_position'][0]).view(1, 1, 1, capacity)
                output = original_forward(**dict(fixed, **tensors, attention_mask=mask))
                return output, torch.stack(output.hidden_states) if output.hidden_states is not None else None
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, hidden = run()
            # Graph replay retains raw pointers to external tensors. Keep the
            # column indices alive as well as the staged inputs and cache.
            state.update(graph=graph, tensors=tensors, output=output, hidden=hidden, columns=columns)
        for key, target in state['tensors'].items():
            value = kwargs[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('talker graph input changed')
            target.copy_(value)
        state['graph'].replay()
        output = state['output']
        hidden = state['hidden']
        h = tuple(hidden.clone().unbind(0)) if hidden is not None else None
        last = h[-1] if h is not None else output.last_hidden_state.clone()
        return BaseModelOutputWithPast(last_hidden_state=last, past_key_values=cache,
                                       hidden_states=h, attentions=output.attentions)

    talker.generate = generate
    decoder.forward = forward
    return state


def configure_fast_runtime(model, packed=True):
    if getattr(model, '_macaq_speech_runtime', False):
        return
    from tts_kernels import use_speech_kernels, use_speech_norms
    if packed:
        use_speech_kernels(model)
    use_speech_norms(model)
    use_predictor_graphs(model)
    use_talker_graphs(model)
    model._macaq_speech_runtime = True
