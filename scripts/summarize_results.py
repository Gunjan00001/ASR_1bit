"""Generate RESULTS.md from the committed, measured result JSONs.

This script performs **no measurement** and invents no numbers. It formats what
is already in ``results/`` and marks missing arms as "not measured". Run it after
the cluster jobs finish:

    python scripts/summarize_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

TRAINING_ARMS = [
    ("baseline.json", "FP32 baseline", "local CPU"),
    ("ptq_binary.json", "1-bit PTQ (attention+FFN)", "local CPU"),
    ("qat_smoke.json", "QAT smoke (attention+FFN, 8 steps)", "local CPU"),
    ("qat_attn.json", "QAT attention-only (Kaggle T4)", "cloud-gpu"),
    ("qat_attn_repro.json", "QAT attention-only reproduction (Kaggle T4)", "cloud-gpu"),
    ("qat_attn_a100.json", "QAT attention-only (A100 re-run)", "cloud-gpu"),
    ("qat_ffn_a100.json", "QAT FFN-only (A100)", "cloud-gpu"),
    ("qat_attn_ffn.json", "QAT attention+FFN (A100)", "cloud-gpu"),
    ("distill_attn_ffn.json", "QAT + distillation attention+FFN, lambda_kd=0.5 (A100)", "cloud-gpu"),
    ("distill_attn_ffn_kd01.json", "QAT + distillation attention+FFN, lambda_kd=0.1 (A100)", "cloud-gpu"),
    ("distillation_smoke.json", "Distillation smoke (8 steps)", "local CPU"),
]


def load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def eval_of(d: dict) -> tuple:
    ev = d.get("eval", {})
    if "wer" in ev:
        return ev.get("wer"), ev.get("cer"), ev.get("rtf")
    return d.get("wer"), d.get("cer"), d.get("rtf")


def training_table() -> str:
    lines = [
        "| Arm | Profile | WER | CER | RTF | Steps | Final loss | Quantized layers | Result file |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for filename, label, profile in TRAINING_ARMS:
        d = load(filename)
        if not d:
            lines.append(f"| {label} | {profile} | — | — | — | — | — | — | not measured |")
            continue
        wer, cer, rtf = eval_of(d)
        steps = d.get("training", {}).get("steps_completed")
        loss = d.get("training", {}).get("final_train_loss")
        replacement = d.get("replacement") or d.get("student", {}).get("replacement", {})
        n_repl = replacement.get("n_replaced", "")
        fmt = lambda v, p=4: f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"
        lines.append(
            f"| {label} | {d.get('compute_profile', profile)} | {fmt(wer)} | {fmt(cer)} | "
            f"{fmt(rtf, 4)} | {steps if steps is not None else '—'} | {fmt(loss, 1)} | "
            f"{n_repl if n_repl else '—'} | `results/{filename}` |"
        )
    return "\n".join(lines)


def size_table() -> str:
    lines = [
        "| Artifact | Layers | Quantized weights | Packed payload (MB) | Non-quantized FP (MB) | FP masters (MB) | Measured artifact (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    files = [
        ("size_report.json", "PTQ attention+FFN"),
        ("qat_attn_size_report.json", "QAT attention-only (Kaggle)"),
        ("qat_attn_repro_size_report.json", "QAT attention-only repro"),
        ("qat_attn_a100_size_report.json", "QAT attention-only (A100)"),
        ("qat_ffn_a100_size_report.json", "QAT FFN-only (A100)"),
        ("qat_attn_ffn_size_report.json", "QAT attention+FFN (A100)"),
        ("distill_attn_ffn_size_report.json", "QAT + distillation attention+FFN, lambda_kd=0.5"),
        ("distill_attn_ffn_kd01_size_report.json", "QAT + distillation attention+FFN, lambda_kd=0.1"),
    ]
    for filename, label in files:
        d = load(filename)
        if not d:
            lines.append(f"| {label} | — | — | — | — | — | not measured |")
            continue
        mb = lambda v: f"{v / 1e6:.1f}" if isinstance(v, (int, float)) else "—"
        lines.append(
            f"| {label} | {d.get('n_bitlinear_layers', '—')} | "
            f"{d.get('quantized_weight_count', '—')} | {mb(d.get('packed_payload_bytes'))} | "
            f"{mb(d.get('non_quantized_fp_param_bytes'))} | "
            f"{mb(d.get('fp_master_bytes_excluded_from_artifact'))} | "
            f"{mb(d.get('packed_artifact_bytes_measured'))} |"
        )
    return "\n".join(lines)


def comparison_section(split: str) -> str:
    path = RESULTS / f"final_comparison_{split}_n256.json"
    d = load(path.name)
    if not d:
        return f"### {split}-clean (256 utterances)\n\nNot measured yet.\n"
    header = (
        f"### {split}-clean (256 utterances)\n\n"
        f"Device: `{d.get('device_name')}` | dataset: `{d.get('dataset')}` | "
        f"greedy CTC | seed {d.get('seed')}\n\n"
    )
    skipped = d.get("skipped", [])
    note = ""
    if skipped:
        note = "\nSkipped arms: " + ", ".join(
            f"`{s['arm']}` ({s['reason']})" for s in skipped
        ) + "\n"
    return header + d.get("table_markdown", "") + note


def packed_section() -> str:
    d = load("packed_inference_benchmark.json")
    if not d:
        return "Not measured yet.\n"
    storage = d.get("storage", {})
    inf = d.get("inference", {})
    kernel = d.get("binary_compute_boundary", {})
    return (
        f"- Packed layers: **{storage.get('n_packed_layers')}**\n"
        f"- Packed weight bytes: **{storage.get('packed_weight_bytes', 0) / 1e6:.2f} MB**\n"
        f"- Non-packed FP bytes: **{storage.get('non_packed_fp_param_bytes', 0) / 1e6:.1f} MB**\n"
        f"- BitLinear forward: **{inf.get('bitlinear_forward_s', float('nan')):.3f} s**; "
        f"Packed forward: **{inf.get('packed_forward_s', float('nan')):.3f} s** "
        f"(slowdown **{inf.get('packed_slowdown_ratio', float('nan')):.2f}×**)\n"
        f"- XNOR+popcount matches float ±1 dot: **{kernel.get('xnor_popcount_matches_float_dot')}**; "
        f"XNOR/float time ratio **{kernel.get('xnor_vs_float_ratio')}**\n"
        f"- Conclusion: {d.get('conclusion')}\n"
    )


def streaming_section() -> str:
    d = load("streaming_demo_qat_attn_ffn.json")
    if not d:
        return "Not measured yet.\n"
    return (
        f"- Sample: `{d.get('sample_id')}`, audio {d.get('audio_seconds', 0):.2f} s, "
        f"chunk {d.get('chunk_s')} s / overlap {d.get('overlap_s')} s, "
        f"{d.get('n_chunks')} chunks\n"
        f"- Chunked WER/CER: **{d.get('chunked_wer')} / {d.get('chunked_cer')}**\n"
        f"- Full-utterance WER/CER: **{d.get('full_utterance_wer')} / {d.get('full_utterance_cer')}**\n"
        f"- Chunked RTF: **{d.get('rtf')}**; scope: {d.get('scope')}\n"
    )


def main() -> int:
    text = f"""# Results — measured artifacts

This file is generated by `scripts/summarize_results.py` from the JSON files in
`results/`. It performs no measurement and fabricates no values. Empty cells /
"not measured" mean exactly that.

Profiles are never mixed: Kaggle T4 results and A100-cluster results are labelled
separately, and each JSON records its own dependency versions.

## 1. Model results

{training_table()}

## 2. Size accounting

{size_table()}

The **measured artifact** is the packed safetensors file (packed 1-bit weights +
FP32 scales + non-quantized FP parameters). FP32 master weights are training-time
state and are excluded from the deploy artifact.

The `RTF` column in §1 comes from each training script's evaluation pass (which
includes model/processor warm-up and therefore looks slower). The warmed-up
benchmark numbers are in §3–§4.

## 3. Final comparison — dev-clean

{comparison_section("dev")}

## 4. Final comparison — test-clean

{comparison_section("test")}

## 5. Packed 1-bit inference (Phase 18)

{packed_section()}

## 6. Chunked streaming (Phase 19)

{streaming_section()}

## 7. Honest reading

- Accuracy numbers are greedy CTC on fixed LibriSpeech subsets; they are not
  comparable to LM-assisted leaderboard numbers.
- A 1-bit weight model is smaller on disk only to the extent that the artifact
  is packed. The runtime in this repository expands weights before `F.linear`,
  so packed storage does **not** imply faster inference.
- The FP32 master weights are retained for QAT; the deploy artifact excludes
  them, but an in-memory training model still holds them.
"""
    out = ROOT / "RESULTS.md"
    out.write_text(text)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
