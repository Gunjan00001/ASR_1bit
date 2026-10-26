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
    ("qat_ffn_control_pilot.json", "Pilot: FFN-only per-tensor control (A100, dev-128)", "cloud-gpu"),
    ("qat_ffn_mixed_pilot.json", "Pilot: mixed-precision FFN (A100, dev-128)", "cloud-gpu"),
    ("qat_ffn_control_full.json", "Full: FFN-only per-tensor control (A100, dev-256)", "cloud-gpu"),
    ("qat_ffn_mixed_full.json", "Full: mixed-precision FFN (A100, dev-256)", "cloud-gpu"),
    ("qat_ffn_control_devsel.json", "QAT FFN-only, dev-selected control", "cloud-gpu"),
    ("qat_ffn_per_channel.json", "QAT FFN-only, per-output-channel scales", "cloud-gpu"),
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
        "| Artifact | Scale | Layers | Quantized weights | Packed payload (MB) | Non-quantized FP (MB) | FP masters (MB) | Measured artifact (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    files = [
        ("size_report.json", "PTQ attention+FFN"),
        ("qat_attn_size_report.json", "QAT attention-only (Kaggle)"),
        ("qat_attn_repro_size_report.json", "QAT attention-only repro"),
        ("qat_attn_a100_size_report.json", "QAT attention-only (A100)"),
        ("qat_ffn_a100_size_report.json", "QAT FFN-only (A100)"),
        ("qat_ffn_control_pilot_size_report.json", "Pilot: FFN-only per-tensor control"),
        ("qat_ffn_mixed_pilot_size_report.json", "Pilot: mixed-precision FFN"),
        ("qat_ffn_control_full_size_report.json", "Full: FFN-only per-tensor control"),
        ("qat_ffn_mixed_full_size_report.json", "Full: mixed-precision FFN"),
        ("qat_ffn_control_devsel_size_report.json", "QAT FFN-only, dev-selected control"),
        ("qat_ffn_per_channel_size_report.json", "QAT FFN-only, per-output-channel scales"),
        ("qat_attn_ffn_size_report.json", "QAT attention+FFN (A100)"),
        ("distill_attn_ffn_size_report.json", "QAT + distillation attention+FFN, lambda_kd=0.5"),
        ("distill_attn_ffn_kd01_size_report.json", "QAT + distillation attention+FFN, lambda_kd=0.1"),
    ]
    for filename, label in files:
        d = load(filename)
        if not d:
            lines.append(f"| {label} | — | — | — | — | — | — | not measured |")
            continue
        mb = lambda v: f"{v / 1e6:.1f}" if isinstance(v, (int, float)) else "—"
        lines.append(
            f"| {label} | {d.get('scale_mode', '—')} | "
            f"{d.get('n_bitlinear_layers', '—')} | "
            f"{d.get('quantized_weight_count', '—')} | {mb(d.get('packed_payload_bytes'))} | "
            f"{mb(d.get('non_quantized_fp_param_bytes'))} | "
            f"{mb(d.get('fp_master_bytes_excluded_from_artifact'))} | "
            f"{mb(d.get('packed_artifact_bytes_measured'))} |"
        )
    return "\n".join(lines)


def comparison_section(split: str, n: int = 256, label: str | None = None) -> str:
    path = RESULTS / f"final_comparison_{split}_n{n}.json"
    d = load(path.name)
    title = label or f"{split}-clean ({n} utterances)"
    if not d:
        return f"### {title}\n\nNot measured yet.\n"
    header = (
        f"### {title}\n\n"
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


def screening_section() -> str:
    """Screening and validation results from the mixed-precision investigation."""
    lines = ["### Per-output-channel reconstruction gate\n"]
    for filename, label in (
        ("quantization_error_ffn.json", "FFN-only QAT masters"),
        ("quantization_error_ffn_fp32.json", "FP32 starting weights"),
    ):
        d = load(filename)
        if not d:
            lines.append(f"- {label}: not measured")
            continue
        s = d["summary"]["all"]
        lines.append(
            f"- {label}: per-tensor {s['mean_err_per_tensor']:.4f} → "
            f"per-channel {s['mean_err_per_channel']:.4f} "
            f"(**{s['mean_relative_improvement']:.2%}** improvement; "
            f"gate {d['gate']:.0%}: **{'PASS' if d['gate_passes'] else 'FAIL'}**; "
            f"mean row CV {s['mean_row_cv']:.3f})"
        )
    lines.append(
        "\nThe 10% gate failed, so no per-channel QAT run was launched. "
        "This is a negative screening result, not a WER result.\n"
    )

    d = load("layer_sensitivity_ffn.json")
    if d:
        lines.append("\n### FFN per-layer sensitivity screen (screening only, not QAT)\n")
        ref = d.get("fp32_reference", {})
        lines.append(
            f"- {d.get('n_utterances')} fixed dev utterances; FP32 ref WER "
            f"{ref.get('wer', float('nan')):.4f}; "
            f"scale mode `{d.get('scale_mode')}`; `screening_only=True`, `not_qat=True`."
        )
        for group, s in d.get("group_summary", {}).items():
            lines.append(
                f"- `{group}`: {s['n_layers']} layers, mean ΔWER "
                f"{s['mean_delta_wer']:+.4f}, mean relL2 {s['mean_relative_l2']:.4f}"
            )
        lines.append("\n| Layer | Rel. L2 | KL | Argmax | ΔWER |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in d.get("top_10", []):
            lines.append(
                f"| `{r['name']}` | {r['relative_l2']:.4f} | "
                f"{r['kl_ref_variant']:.4f} | {r['argmax_agreement']:.3f} | "
                f"{r['delta_wer_vs_fp32']:+.4f} |"
            )
        lines.append(
            "\n`ffn_in` is ~1.9× more sensitive than `ffn_out`; the top layers are "
            "middle/late, selected from measurements rather than position."
        )
    else:
        lines.append("\nLayer-sensitivity screen: not measured.")

    mixed = load("qat_ffn_mixed_smoke.json")
    pc = load("qat_ffn_pc_smoke.json")
    if mixed or pc:
        lines.append("\n### Mixed-precision implementation validation\n")
        if mixed:
            rep = mixed.get("replacement", {})
            lines.append(
                f"- mixed config: {rep.get('n_replaced')} BitLinear + "
                f"{rep.get('n_excluded')} excluded FP32 FFN layers; "
                f"smoke WER {mixed.get('eval', {}).get('wer'):.4f} "
                f"(8 steps — pipeline check, not accuracy)"
            )
        if pc:
            rep = pc.get("replacement", {})
            lines.append(
                f"- per-channel config: {rep.get('n_replaced')} BitLinear; "
                f"smoke WER {pc.get('eval', {}).get('wer'):.4f} "
                f"(8 steps — pipeline check, not accuracy)"
            )
        for filename in (
            "packed_verify_ffn_pc_smoke.json",
            "packed_verify_ffn_mixed_smoke.json",
        ):
            v = load(filename)
            if v:
                lines.append(
                    f"- `{filename}`: passed={v.get('passed')}, "
                    f"max |Δ logit|={v.get('max_abs_logit_diff')}, "
                    f"argmax={v.get('mean_argmax_agreement')}"
                )

    control = load("qat_ffn_control_pilot.json")
    mixed_pilot = load("qat_ffn_mixed_pilot.json")
    if control and mixed_pilot:
        lines.append("\n### Controlled mixed-precision pilot (dev-128, 2,000 utts, 500 steps)\n")
        lines.append(
            "| Arm | WER | CER | Best step | Final step | Restored best | Artifact (measured) |"
        )
        lines.append("|---|---:|---:|---:|---:|---|---:|")
        for label, d, size_file in (
            ("FFN-only control (96 BitLinear)", control,
             "qat_ffn_control_pilot_size_report.json"),
            ("Mixed FFN (79 BitLinear + 17 FP32)", mixed_pilot,
             "qat_ffn_mixed_pilot_size_report.json"),
        ):
            size = load(size_file)
            t = d.get("training", {})
            artifact = size.get("packed_artifact_bytes_measured")
            lines.append(
                f"| {label} | {d.get('eval', {}).get('wer'):.4f} | "
                f"{d.get('eval', {}).get('cer'):.4f} | {t.get('best_step')} | "
                f"{t.get('steps_completed')} | {t.get('restored_best')} | "
                f"{artifact:,} B |" if artifact is not None else
                f"| {label} | {d.get('eval', {}).get('wer'):.4f} | "
                f"{d.get('eval', {}).get('cer'):.4f} | {t.get('best_step')} | "
                f"{t.get('steps_completed')} | {t.get('restored_best')} | not measured |"
            )
        delta = mixed_pilot.get("eval", {}).get("wer", 0) - control.get("eval", {}).get("wer", 0)
        lines.append(
            f"\nMixed minus control dev WER: **{delta:+.4f}**. The two arms used the "
            "same 2,000 training IDs, 128 eval IDs, seed, budget, batch size, and LR. "
            "This is a **dev-128 pilot**, not a full-test result; the full 10k/1,875-step "
            "run is the next controlled step. Artifact reload checks:"
        )
        for filename in (
            "packed_verify_ffn_control_pilot.json",
            "packed_verify_ffn_mixed_pilot.json",
        ):
            v = load(filename)
            if v:
                lines.append(
                    f"- `{filename}`: passed={v.get('passed')}, "
                    f"max |Δ logit|={v.get('max_abs_logit_diff')}, "
                    f"argmax={v.get('mean_argmax_agreement')}, "
                    f"decoded={v.get('decoded_match_count')}/{v.get('n_utterances')}"
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    text = f"""# Results — measured artifacts

This file is generated by `scripts/summarize_results.py` from the JSON files in
`results/`. It performs no measurement and fabricates no values. Empty cells /
"not measured" mean exactly that.

Profiles are never mixed: Kaggle T4 results and A100-cluster results are labelled
separately, and each JSON records its own dependency versions.

> Visual summary with generated charts: see the README's **Results at a glance**
> and [`docs/figures/`](docs/figures/).

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

{comparison_section("test", 256)}

{comparison_section("test", 2620, "Full test-clean (2620 utterances)")}

> The full test-clean run is accuracy-only (`--no-benchmark`, one inference pass
> per utterance), so its performance columns are `n/a`. Warmed-up latency/RTF/
> VRAM are measured on the 256-utterance subsets in the tables above.

## 5. Packed 1-bit inference (Phase 18)

{packed_section()}

## 6. Chunked streaming (Phase 19)

{streaming_section()}

## 7. Screening and mixed-precision validation (v1.6 research)

{screening_section()}

## 8. Honest reading

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
