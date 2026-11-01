# 2-bit FFN

Accuracy winner. 2-bit FFN with the fixed 17 FP32 FFN exclusions and all attention FP32.

- Full test-clean WER: **2.58%**
- Dev-256 WER: `0.024605920799692427`
- Artifact: `1,131,269,460 B`
- SHA-256: `e8e138050f2252a3b9aa327aec87cbe3340f244509ab2b6953d491bd8881c856`

## Runtime

This model is served through `onebit_asr.serving.ModelManager`.
Packed weights are expanded to FP32 before matrix multiplication;
the artifact is a storage format, not an inference accelerator.

The deploy weight file is tracked with Git LFS in the hosted
repository. Local development falls back to the verified v2
artifact at `checkpoints/qat_ffn_mixed_2bit_full_packed_v2/model_packed.safetensors`.
