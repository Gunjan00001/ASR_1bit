"""One-off verification: the refactored evaluator reproduces baseline predictions.

Compares hypotheses produced by ``onebit_asr.evaluation.evaluate.evaluate_model``
against the per-utterance hypotheses already recorded in results/baseline.json
for the same fixed subset IDs. Read-only: baseline.json is not modified.

Usage:
    python scripts/verify_eval_equivalence.py [--n 8]
"""

import argparse
import json
from pathlib import Path

from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.evaluation.evaluate import evaluate_model

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "results" / "baseline.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="number of utterances to re-check")
    args = parser.parse_args()

    baseline = json.loads(BASELINE.read_text())
    by_id = {u["id"]: u for u in baseline["utterances"]}
    cfg = load_config(ROOT / "configs" / "baseline.yaml")
    eval_cfg = cfg["eval"]

    samples = load_dev_subset(subset_size=eval_cfg["subset_size"], seed=eval_cfg["seed"])
    subset = samples[: args.n]

    processor = Wav2Vec2Processor.from_pretrained(cfg["model"])
    model = Wav2Vec2ConformerForCTC.from_pretrained(cfg["model"])
    model.eval()

    result = evaluate_model(model, processor, subset, verbose=False)

    mismatches = []
    for utt in result["utterances"]:
        expected = by_id[utt["id"]]
        if utt["hypothesis"] != expected["hypothesis"]:
            mismatches.append(utt["id"])

    print(f"re-checked: {len(result['utterances'])} utterances")
    print(f"sample WER: {result['wer']:.4f}")
    print(f"hypothesis mismatches: {len(mismatches)}")
    if mismatches:
        print("  mismatched ids:", mismatches)
        raise SystemExit("EQUIVALENCE CHECK FAILED")
    print("EQUIVALENCE CHECK PASSED: refactored evaluator matches baseline.json")


if __name__ == "__main__":
    main()
