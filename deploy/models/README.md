# Selected model packages

Each directory contains the model configuration, tokenizer/processor files,
packed-artifact manifest, metadata, and SHA-256 checksums. The large
`model_packed.safetensors` files are intended for Git LFS (see the repository
`.gitattributes`).

| Model | Full test WER | Historical artifact | Standalone v2 artifact |
|---|---:|---:|---:|
| 2-bit FFN | 2.58% | 1,131,062,436 B | 1,131,269,460 B |
| Strategy A | 2.74% | 989,509,572 B | 989,716,588 B |
| FP32 reference | 1.93% | 2,373,507,712 B | Hugging Face download |

For local development, the demo resolves the verified v2 artifacts in
`checkpoints/*_packed_v2/` when a deploy weight file is absent. For a clean
clone, fetch the LFS files:

```bash
git lfs install
git lfs pull --include="deploy/models/**"
```

The frontend reports an explicit error if a requested artifact is unavailable;
it never substitutes another model silently.
