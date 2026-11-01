# Strategy A (sub-1 GB)

Compression winner. 36 attention projections and 79 FFN layers 2-bit; 60 attention + 17 FFN projections FP32; below 1 GB.

- Full test-clean WER: **2.74%**
- Dev-256 WER: `0.027297193387158785`
- Artifact: `989,716,588 B`
- SHA-256: `182c62d18ef00af897a21506848f1ec8246070596c8611988294bc7f80293ca9`

## Runtime

This model is served through `onebit_asr.serving.ModelManager`.
Packed weights are expanded to FP32 before matrix multiplication;
the artifact is a storage format, not an inference accelerator.

The deploy weight file is tracked with Git LFS in the hosted
repository. Local development falls back to the verified v2
artifact at `checkpoints/attn_2bit_on_2bitffn_A_full_packed_v2/model_packed.safetensors`.
