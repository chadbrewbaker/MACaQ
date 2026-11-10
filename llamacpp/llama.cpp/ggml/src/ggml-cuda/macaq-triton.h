#pragma once

// Host-only driver interface to ahead-of-time compiled Triton kernels.
void ggml_macaq_triton_prepare();
bool ggml_macaq_triton_available(int block, bool quantized);
void ggml_macaq_triton_launch(const void * x, const void * scales, void * y, void * packed,
                             int width, int rows, int padded, int block,
                             bool quantized, void * stream);

bool ggml_macaq_triton_moe_available(bool quantized);
void ggml_macaq_triton_moe_launch(const void * x, const void * scales, const void * ids,
                                 void * y, void * packed, int width, int tokens, int routes,
                                 int input_routes, int padded, int id_stride, int experts,
                                 bool quantized, void * stream);
