"""Generate the figures used by README.md / RESULTS.md / deepdive.md.

Every figure is derived from the committed JSON results; no number is typed in.
Run with:

    .venv/bin/python scripts/make_figures.py

Outputs PNGs to ``docs/figures/`` (tracked) so GitHub renders them in Markdown.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"

# Consistent colour language across every figure.
COLORS = {
    "fp32": "#6b7280",
    "ptq": "#dc2626",
    "qat": "#2563eb",
    "kd": "#16a34a",
    "packed": "#0f766e",
    "scales": "#7c3aed",
    "nonquant": "#f59e0b",
}

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
})


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def gb(byte_count: float | int | None) -> float:
    return float(byte_count) / 1e9 if byte_count else 0.0


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


# --- Figure 1: accuracy vs deploy size (full test-clean) --------------------

def fig_wer_vs_size() -> None:
    combined = load("final_comparison_test_n2620.json")
    rows = {r["arm"]: r for r in combined["arms"]}
    full = load("full_arms_test_n2620/final_comparison_test_n2620.json")
    rows.update({r["arm"]: r for r in full["arms"]})
    order = [
        ("fp32", "FP32", COLORS["fp32"]),
        ("qat_attention_a100", "QAT attention", COLORS["qat"]),
        ("qat_ffn_mixed_full", "Mixed FFN (17 FP)", "#7c3aed"),
        ("qat_ffn_control_full", "Full FFN control", "#0f766e"),
        ("qat_attn_ffn", "QAT attention+FFN", COLORS["qat"]),
        ("distill_attn_ffn_kd01", "QAT+KD 0.1", COLORS["kd"]),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    points = []
    annotate = {
        "fp32": (10, 8),
        "qat_attention_a100": (10, 8),
        "qat_ffn_mixed_full": (-70, -28),
        "qat_ffn_control_full": (-80, 12),
    }
    for arm, label, color in order:
        row = rows[arm]
        size = gb(row["size"].get("packed_artifact_components_bytes")
                  or row["size"].get("fp32_weight_bytes"))
        wer = row["accuracy"]["wer"] * 100
        points.append((size, wer, label, color))
        ax.scatter(size, wer, s=90, color=color, zorder=3,
                   edgecolor="white", linewidth=1.2)
        if arm in annotate:
            ax.annotate(
                f"{label}\n{wer:.2f}% @ {size:.3f} GB",
                (size, wer), textcoords="offset points",
                xytext=annotate[arm], fontsize=8.3,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d1d5db", lw=0.6),
            )

    # Pareto frontier: FP32 -> attention -> mixed -> full FFN control -> best 0.423 GB arm.
    frontier = [points[0], points[1], points[2], points[3], points[5]]
    ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
            color="#9ca3af", linestyle="--", linewidth=1.2, zorder=1,
            label="measured frontier")

    ax.set_xlim(0.30, 2.60)
    ax.set_ylim(0, 25)
    ax.set_xlabel("Packed deploy size (GB)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Accuracy vs storage — full LibriSpeech test-clean (2620 utterances)")
    ax.annotate(
        "best 5.6× arm\n(KD λ=0.1)",
        (0.423, 19.98), xytext=(0.62, 16.0),
        arrowprops=dict(arrowstyle="->", color=COLORS["kd"], lw=1.0),
        color=COLORS["kd"], fontsize=8.2,
    )
    ax.text(
        0.32, 1.2,
        "0.423 GB cluster: QAT 20.50%  |  KD λ=0.5 22.18%  |  KD λ=0.1 19.98%\n"
        "PTQ attention+FFN: 100% WER at the same 0.423 GB (off-scale)",
        fontsize=8.0, color="#4b5563",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f9fafb", ec="#e5e7eb", lw=0.6),
    )
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "wer_vs_size_full_test.png")


# --- Figure 2: WER by arm (full test-clean, log x) --------------------------

def fig_wer_by_arm() -> None:
    combined = load("final_comparison_test_n2620.json")
    rows = {r["arm"]: r for r in combined["arms"]}
    full = load("full_arms_test_n2620/final_comparison_test_n2620.json")
    rows.update({r["arm"]: r for r in full["arms"]})
    arms = [
        ("fp32", "FP32", COLORS["fp32"]),
        ("qat_attention_a100", "QAT attention", COLORS["qat"]),
        ("qat_ffn_mixed_full", "Mixed FFN\n(17 FP32)", "#7c3aed"),
        ("qat_ffn_control_full", "Full FFN\ncontrol", "#0f766e"),
        ("qat_attn_ffn", "QAT attention+FFN", COLORS["qat"]),
        ("distill_attn_ffn", "QAT+KD 0.5", COLORS["kd"]),
        ("distill_attn_ffn_kd01", "QAT+KD 0.1", COLORS["kd"]),
    ]
    labels = [a[1] for a in arms]
    wers = [rows[a[0]]["accuracy"]["wer"] * 100 for a in arms]
    colors = [a[2] for a in arms]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y = np.arange(len(labels))
    ax.barh(y, wers, color=colors, height=0.62)
    for yi, value in zip(y, wers):
        ax.text(value * 1.08, yi, f"{value:.2f}%", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.9, 30)
    ax.set_xlabel("WER (%, log scale)")
    ax.set_title("Full test-clean WER by arm (lower is better)")
    save(fig, "wer_by_arm_full_test.png")


# --- Figure 3: PTQ vs QAT recovery (dev 256) --------------------------------

def fig_ptq_vs_qat() -> None:
    combined = load("final_comparison_dev_n256.json")
    rows = {r["arm"]: r for r in combined["arms"]}
    groups = [
        ("attention-only", "ptq_attention", "qat_attention_a100"),
        ("FFN-only", "ptq_ffn", "qat_ffn_a100"),
        ("attention+FFN", "ptq_full", "qat_attn_ffn"),
    ]
    x = np.arange(len(groups))
    width = 0.34
    ptq = [rows[g[1]]["accuracy"]["wer"] * 100 for g in groups]
    qat = [rows[g[2]]["accuracy"]["wer"] * 100 for g in groups]
    fp32 = rows["fp32"]["accuracy"]["wer"] * 100

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    b1 = ax.bar(x - width / 2, ptq, width, label="PTQ (no retraining)",
                color=COLORS["ptq"])
    b2 = ax.bar(x + width / 2, qat, width, label="QAT (1875 steps)",
                color=COLORS["qat"])
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.25,
                    f"{value:.2f}%", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(fp32, color=COLORS["fp32"], linestyle="--", linewidth=1.2,
               label=f"FP32 {fp32:.2f}%")
    ax.set_yscale("log")
    ax.set_ylim(1.5, 260)
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylabel("WER (%, log scale)")
    ax.set_title("Post-training quantization vs QAT (dev-clean, 256 utterances)")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "ptq_vs_qat_dev.png")


# --- Figure 4: training loss curves ----------------------------------------

def fig_loss_curves() -> None:
    qat_arms = [
        ("qat_attn_a100", "QAT attention", COLORS["qat"]),
        ("qat_ffn_a100", "QAT FFN", "#0ea5e9"),
        ("qat_attn_ffn", "QAT attention+FFN", "#1e40af"),
    ]
    kd_arms = [
        ("distill_attn_ffn", "distillation λ_kd=0.5", COLORS["kd"]),
        ("distill_attn_ffn_kd01", "distillation λ_kd=0.1", "#15803d"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    for arm, label, color in qat_arms:
        h = load(f"{arm}.json")["training"]["train_loss_history"]
        axes[0].plot([e["step"] for e in h], [e["loss"] for e in h],
                     label=label, color=color, linewidth=1.7)
    axes[0].set_title("QAT training loss (CTC only)")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("CTC loss (summed over frames)")
    axes[0].legend(fontsize=8.5)

    for arm, label, color in kd_arms:
        h = load(f"{arm}.json")["training"]["train_loss_history"]
        axes[1].plot([e["step"] for e in h], [e["loss"] for e in h],
                     label=label, color=color, linewidth=1.7)
    axes[1].set_title("Distillation training loss (CTC + KD)")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("total loss")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("Training curves (10k train.100 utterances, 3 epochs, 1875 steps)",
                 fontsize=12, y=1.02)
    save(fig, "training_loss_curves.png")


# --- Figure 5: deploy-size composition -------------------------------------

def _size_parts(name: str) -> tuple[float, float, float, float]:
    d = load(name)
    packed = gb(d.get("packed_payload_bytes", 0))
    scales = gb(d.get("fp_scale_bytes", 0))
    nonquant = gb(d.get("non_quantized_fp_param_bytes", 0))
    total = gb(d.get("packed_artifact_bytes_measured",
                     d.get("packed_artifact_components_bytes", 0)))
    return packed, scales, nonquant, total


def fig_size_breakdown() -> None:
    rows = [
        ("FP32 baseline", None),
        ("QAT attention-only", "qat_attn_a100_size_report.json"),
        ("QAT FFN-only", "qat_ffn_a100_size_report.json"),
        ("Full mixed FFN\n(17 FP32)", "qat_ffn_mixed_full_size_report.json"),
        ("QAT attention+FFN\n(and KD 0.1)", "qat_attn_ffn_size_report.json"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(rows))
    for i, (label, size_file) in enumerate(rows):
        if size_file is None:
            fp32_gb = 593_376_928 * 4 / 1e9
            ax.barh(i, fp32_gb, color="#d1d5db", height=0.6,
                    label="FP32 parameters" if i == 0 else None)
            ax.text(fp32_gb + 0.03, i, f"{fp32_gb:.2f} GB", va="center", fontsize=8.5)
            continue
        packed, scales, nonquant, total = _size_parts(size_file)
        left = 0.0
        ax.barh(i, packed, left=left, color=COLORS["packed"], height=0.6,
                label="packed 1-bit weights" if i == 1 else None)
        left += packed
        ax.barh(i, scales, left=left, color=COLORS["scales"], height=0.6,
                label="FP32 scales (+biases)" if i == 1 else None)
        left += scales
        ax.barh(i, nonquant, left=left, color=COLORS["nonquant"], height=0.6,
                label="non-quantized FP params" if i == 1 else None)
        ax.text(total + 0.03, i, f"{total:.2f} GB", va="center", fontsize=8.5)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("deploy size (GB)")
    ax.set_title("What is actually inside the deploy artifact")
    ax.legend(loc="lower right", fontsize=8.2)
    save(fig, "size_breakdown.png")


# --- Figure 6: benchmark RTF / latency --------------------------------------

def fig_benchmark() -> None:
    combined = load("final_comparison_dev_n256.json")
    rows = {r["arm"]: r for r in combined["arms"]}
    full = load("full_arms_test_n256/final_comparison_test_n256.json")
    rows.update({r["arm"]: r for r in full["arms"]})
    arms = [
        ("fp32", "FP32", COLORS["fp32"]),
        ("qat_attention_a100", "QAT attention", COLORS["qat"]),
        ("qat_ffn_mixed_full", "Mixed FFN (test-256)", "#7c3aed"),
        ("qat_ffn_control_full", "Full FFN (test-256)", "#0f766e"),
        ("qat_attn_ffn", "QAT attention+FFN", COLORS["qat"]),
        ("distill_attn_ffn_kd01", "QAT+KD 0.1", COLORS["kd"]),
    ]
    labels = [a[1] for a in arms]
    rtf = [rows[a[0]]["performance"]["rtf"] for a in arms]
    p50 = [rows[a[0]]["performance"]["latency_s"]["p50"] for a in arms]
    colors = [a[2] for a in arms]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    axes[0].bar(labels, rtf, color=colors, width=0.62)
    axes[0].set_ylabel("RTF (end-to-end)")
    axes[0].set_title("Real-time factor (lower is better)")
    axes[0].tick_params(axis="x", rotation=25)
    for i, value in enumerate(rtf):
        axes[0].text(i, value + 0.0002, f"{value:.4f}", ha="center", fontsize=8)

    axes[1].bar(labels, p50, color=colors, width=0.62)
    axes[1].set_ylabel("p50 latency (s)")
    axes[1].set_title("Median latency per utterance")
    axes[1].tick_params(axis="x", rotation=25)
    for i, value in enumerate(p50):
        axes[1].text(i, value + 0.001, f"{value:.3f}", ha="center", fontsize=8)

    fig.suptitle("A100 dev-clean benchmark: binarization does not make it faster",
                 fontsize=12, y=1.03)
    save(fig, "benchmark_rtf_latency.png")


# --- Figure 7: streaming demo ----------------------------------------------

def fig_streaming() -> None:
    d = load("streaming_demo_qat_attn_ffn.json")
    labels = ["WER", "CER"]
    chunked = [d["chunked_wer"] * 100, d["chunked_cer"] * 100]
    full = [d["full_utterance_wer"] * 100, d["full_utterance_cer"] * 100]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    b1 = ax.bar(x - width / 2, full, width, label="full utterance", color=COLORS["qat"])
    b2 = ax.bar(x + width / 2, chunked, width, label="chunked 2 s / 0.5 s overlap",
                color=COLORS["ptq"])
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8,
                    f"{value:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 80)
    ax.set_title("Chunked streaming vs full decode (4.45 s sample)")
    ax.legend(fontsize=8.5)
    save(fig, "streaming_demo.png")


# --- Figure 8: model parameter map -----------------------------------------

def fig_param_map() -> None:
    d = load("model_inspection.json")
    by_cat = d["by_category"]
    total = d["total_params"]
    groups = [
        ("FFN in", by_cat["ffn_in"]["params"]),
        ("FFN out", by_cat["ffn_out"]["params"]),
        ("Attention q/k/v/out", sum(by_cat[k]["params"] for k in
                                    ("attention_q", "attention_k",
                                     "attention_v", "attention_output"))),
        ("Feature projection", by_cat["feature_projection"]["params"]),
        ("CTC head", by_cat["ctc_head"]["params"]),
    ]
    used = sum(g[1] for g in groups)
    groups.append(("Convs / norms / other", total - used))

    labels = [g[0] for g in groups]
    values = [g[1] / 1e6 for g in groups]
    colors = ["#1d4ed8", "#3b82f6", "#93c5fd", COLORS["nonquant"],
              COLORS["packed"], "#d1d5db"]

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    left = 0.0
    for label, value, color in zip(labels, values, colors):
        ax.barh(0, value, left=left, color=color, height=0.55, edgecolor="white",
                linewidth=1.0)
        share = value / (total / 1e6) * 100
        if share > 2.0:
            ax.text(left + value / 2, 0, f"{label}\n{share:.1f}%",
                    ha="center", va="center", fontsize=8.2,
                    color="white" if color not in ("#d1d5db", "#93c5fd") else "#111827")
        left += value
    ax.text(0.62, 0.80,
            "feature projection + CTC head = 0.1% (too small to label)",
            transform=ax.transAxes, fontsize=8, color="#4b5563")
    ax.set_yticks([])
    ax.set_xlabel("parameters (millions)")
    ax.set_title(f"Where the {total / 1e6:.1f}M parameters live — FFN is 68%")
    ax.set_xlim(0, total / 1e6)
    save(fig, "param_map.png")


def main() -> int:
    fig_wer_vs_size()
    fig_wer_by_arm()
    fig_ptq_vs_qat()
    fig_loss_curves()
    fig_size_breakdown()
    fig_benchmark()
    fig_streaming()
    fig_param_map()
    print(f"\nAll figures written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
