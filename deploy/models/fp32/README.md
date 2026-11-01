# FP32 reference

Unquantized reference. Not uploaded; downloaded from Hugging Face at runtime if permitted.

- Full test-clean WER: **1.93%**
- Dev-256 WER: `0.020184544405997693`
- Artifact: `2,373,507,712 B`
- SHA-256: `97bb9761fb71ec1225100bc81ccf7d002e0d0ba3d0604c1fd2dbda7d7d491f1d`

## Runtime

This model is served through `onebit_asr.serving.ModelManager`.
Packed weights are expanded to FP32 before matrix multiplication;
the artifact is a storage format, not an inference accelerator.
