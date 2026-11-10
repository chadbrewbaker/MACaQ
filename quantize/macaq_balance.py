"""Shared MACAQ D1 calculation from activation diagonal statistics."""


def balance_pre_scale_from_diag(W, sum_x2, n_tokens, alpha=0.75):
    w_std = W.float().std(dim=0).clamp_min(1e-8)
    act_rms = (sum_x2.float() / max(float(n_tokens), 1.0)).clamp_min(1e-12).sqrt()
    return (w_std.pow(alpha) / act_rms.pow(1.0 - alpha)).clamp(1e-4, 1e4)
