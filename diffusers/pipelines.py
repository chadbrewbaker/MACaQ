"""Explicit pipeline selection for the supported diffusion adapters."""


def select_pipeline(config, family, image_conditioned=False):
    from diffusers import Flux2KleinPipeline, WanPipeline, WanImageToVideoPipeline, StableAudioPipeline

    if config.get("transformer_2") not in (None, [None, None]):
        raise ValueError("dual-transformer pipelines need a separate adapter")
    declared = config.get("_class_name")
    if family == "stable-audio":
        if image_conditioned or declared != "StableAudioPipeline":
            raise ValueError("expected a text-to-audio StableAudioPipeline")
        return StableAudioPipeline
    if family == "flux2-klein":
        if image_conditioned or declared != "Flux2KleinPipeline":
            raise ValueError("expected a text-to-image Flux2KleinPipeline")
        return Flux2KleinPipeline
    if family != "wan":
        raise ValueError("unsupported diffusion family")
    if image_conditioned:
        # The official TI2V-5B repo declares WanPipeline for T2V. Its 48-channel,
        # expanded-timestep transformer also supports the I2V pipeline.
        if declared == "WanImageToVideoPipeline" or (
            declared == "WanPipeline" and config.get("expand_timesteps") is True
        ):
            return WanImageToVideoPipeline
        raise ValueError("this Wan source does not declare image conditioning support")
    if declared != "WanPipeline":
        raise ValueError("expected a text-to-video WanPipeline")
    return WanPipeline


def load_pipeline(model, family, image_conditioned=False, revision=None, transformer=None):
    import torch
    from diffusers import DiffusionPipeline, AutoencoderKLWan

    config = DiffusionPipeline.load_config(model, revision=revision)
    cls = select_pipeline(config, family, image_conditioned)
    components = {} if transformer is None else {"transformer": transformer}
    if family == "wan":
        # The official Wan recipe keeps the VAE in FP32 for encode/decode.
        components["vae"] = AutoencoderKLWan.from_pretrained(
            model, subfolder="vae", revision=revision, torch_dtype=torch.float32)
    return cls.from_pretrained(model, revision=revision, torch_dtype=torch.bfloat16, **components)
