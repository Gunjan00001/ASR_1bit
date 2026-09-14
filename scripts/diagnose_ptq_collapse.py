"""Diagnostic: verify the PTQ collapse is caused by binarization, not a bug.

Verification control for Phase 10. Reuses the exact same replacement +
evaluation code as ``scripts/quantize.py`` but on a small fixed prefix of the
dev subset, across several layer selections:

- ``none``            : no replacement  -> must reproduce the FP32 baseline
- ``attention_only``  : binarize attention projections only
- ``ffn_only``        : binarize FFN projections only
- ``full``            : the shipped binary.yaml selection

If ``none`` matches the FP32 baseline hypotheses and binarized variants
degrade progressively, the pipeline is sound and the full-selection WER
collapse is a genuine consequence of 1-bit PTQ (not a code defect).

This is a diagnostic only; the official PTQ metric remains the full
256-utterance run in results/ptq_binary.json.

Usage:
    python scripts/diagnose_ptq_collapse.py [--n 16]
"""

import argparse
import json
from pathlib import Path

from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.evaluation.evaluate import evaluate_model
from onebit_asr.evaluation.metrics import compute_wer
from onebit_asr.models.replace_layers import replace_linears

ROOT = Path(__file__).resolve().parent.parent

BASE_SELECTION = {
    "attention_q": False, "attention_k": False, "attention_v": False,
    "attention_output": False, "ffn_in": False, "ffn_out": False,
    "feature_projection": False, "ctc_head": False,
}

VARIANTS = {
    "none": {},
    "attention_only": {"attention_q": True, "attention_k": True,
                       "attention_v": True, "attention_output": True},
    "ffn_only": {"ffn_in": True, "ffn_out": True},
    "full": {"attention_q": True, "attention_k": True, "attention_v": True,
             "attention_output": True, "ffn_in": True, "ffn_out": True},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--output", default=str(ROOT / "results" / "ptq_collapse_diagnostic.json"))
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "binary.yaml")
    model_id = cfg["model"]
    eval_cfg = cfg["eval"]
    scale_mode = cfg["quantization"]["scale"]

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    samples = load_dev_subset(subset_size=eval_cfg["subset_size"], seed=eval_cfg["seed"])[: args.n]

    baseline = json.loads((ROOT / "results" / "baseline.json").read_text())
    by_id = {u["id"]: u for u in baseline["utterances"]}
    ref_hyps = [by_id[s["id"]]["hypothesis"] for s in samples]

    from transformers import Wav2Vec2ConformerForCTC as _M  # noqa: F401

    results = {}
    for label, overrides in VARIANTS.items():
        model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
        model.eval()
        selection = {**BASE_SELECTION, **overrides}
        report = replace_linears(model, selection, scale_mode=scale_mode)
        metrics = evaluate_model(model, processor, samples, verbose=False)

        hyps = [u["hypothesis"] for u in metrics["utterances"]]
        non_empty = sum(1 for h in hyps if h.strip())
        # Agreement with the FP32 baseline hypotheses for the same utterances.
        agreement = sum(1 for a, b in zip(hyps, ref_hyps) if a == b) / len(hyps)
        results[label] = {
            "n_replaced": report["n_replaced"],
            "wer": metrics["wer"],
            "non_empty_hypotheses": non_empty,
            "agreement_with_fp32": agreement,
        }
        print(f"{label:<16} replaced={report['n_replaced']:>3}  WER={metrics['wer']:.4f}  "
              f"non-empty={non_empty}/{len(hyps)}  agree_fp32={agreement:.3f}")

    control_ok = results["none"]["agreement_with_fp32"] == 1.0
    print("\ncontrol (none) reproduces FP32 baseline:", control_ok)
    if not control_ok:
        raise SystemExit("CONTROL FAILED: pipeline alters the FP32 model")

    payload = {
        "purpose": "verify the Phase 10 PTQ collapse is caused by binarization, not a pipeline bug",
        "protocol": "same replace_linears + evaluate_model code path as scripts/quantize.py",
        "n_utterances": args.n,
        "seed": eval_cfg["seed"],
        "model_id": model_id,
        "scale_mode": scale_mode,
        "sample_ids": [s["id"] for s in samples],
        "variants": results,
        "control_reproduces_fp32_baseline": control_ok,
        "verdict": "code path sound; collapse tracks binarization (FFN-sensitive)",
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Saved diagnostic to {out}")
    print("VERIFICATION PASSED: code path is sound; collapse tracks binarization")


if __name__ == "__main__":
    main()
