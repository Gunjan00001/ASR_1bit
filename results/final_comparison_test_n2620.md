| Arm | WER | CER | RTF | p50 (s) | p95 (s) | utt/s | Peak RSS (MB) | Peak VRAM (GB) | Deploy size (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 0.0193 | 0.0050 | n/a | n/a | n/a | n/a | n/a | n/a | 2.374 |
| qat_attention_a100 | 0.0449 | 0.0137 | n/a | n/a | n/a | n/a | n/a | n/a | 1.983 |
| qat_ffn_a100 | 0.1134 | 0.0325 | n/a | n/a | n/a | n/a | n/a | n/a | 0.813 |
| qat_attn_ffn | 0.2050 | 0.0639 | n/a | n/a | n/a | n/a | n/a | n/a | 0.423 |
| distill_attn_ffn | 0.2218 | 0.0709 | n/a | n/a | n/a | n/a | n/a | n/a | 0.423 |
| distill_attn_ffn_kd01 | 0.1998 | 0.0636 | n/a | n/a | n/a | n/a | n/a | n/a | 0.423 |

