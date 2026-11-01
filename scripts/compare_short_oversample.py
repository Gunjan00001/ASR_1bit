"""Compare the short-utterance pilot against Strategy A on paired dev-256."""

from __future__ import annotations

import json
from pathlib import Path

import jiwer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "results/ctc_decoding_A_full_dev256.json"
PILOT = ROOT / "results/short_oversample_pilot_dev256/final_attn_2bit_A_short_oversample_pilot_dev_n256.json"
PILOT_RESULT = ROOT / "results/attn_2bit_A_short_oversample_pilot.json"
PILOT_SIZE = ROOT / "results/attn_2bit_A_short_oversample_pilot_size_report.json"
PILOT_VERIFY = ROOT / "results/packed_verify_attn_2bit_A_short_oversample_pilot.json"
OUT = ROOT / "results/short_oversample_pilot_comparison.json"
SHORT_WORDS = 9
LONG_REGRESSION_LIMIT_PP = 0.10


def rows_from_baseline() -> list[dict]:
    payload = json.loads(BASELINE.read_text())
    return [
        {
            "id": u["id"],
            "reference": u["reference"],
            "hypothesis": u["hypothesis"],
            "audio_s": u["audio_s"],
        }
        for u in payload["per_utterance"]
    ]


def rows_from_pilot() -> tuple[list[dict], dict]:
    payload = json.loads(PILOT.read_text())
    accuracy = payload["accuracy"]
    return (
        [
            {
                "id": u["id"],
                "reference": u["reference"],
                "hypothesis": u["hypothesis"],
                "audio_s": u["audio_s"],
            }
            for u in accuracy["utterances"]
        ],
        accuracy,
    )


def summarize(rows: list[dict]) -> dict:
    words = chars = 0
    substitutions = deletions = insertions = hits = 0
    refs, hyps = [], []
    for row in rows:
        ref, hyp = row["reference"], row["hypothesis"]
        w = jiwer.process_words([ref], [hyp])
        c = jiwer.process_characters([ref], [hyp])
        words += w.hits + w.substitutions + w.deletions
        chars += c.hits + c.substitutions + c.deletions
        substitutions += w.substitutions
        deletions += w.deletions
        insertions += w.insertions
        hits += w.hits
        refs.append(ref)
        hyps.append(hyp)
    word_chars = jiwer.process_characters(refs, hyps)
    word_errors = substitutions + deletions + insertions
    return {
        "n_utterances": len(rows),
        "words": words,
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "word_errors": word_errors,
        "wer": word_errors / words if words else 0.0,
        "cer": float(word_chars.cer),
        "characters": chars,
    }


def slice_stats(rows: list[dict], short: bool) -> dict:
    selected = [
        row for row in rows
        if (len(row["reference"].split()) <= SHORT_WORDS) == short
    ]
    out = summarize(selected)
    out["short"] = short
    return out


def main() -> int:
    base_rows = rows_from_baseline()
    pilot_rows, pilot_accuracy = rows_from_pilot()
    base_ids = {row["id"] for row in base_rows}
    pilot_ids = {row["id"] for row in pilot_rows}
    if base_ids != pilot_ids or len(base_rows) != len(pilot_rows):
        raise SystemExit("baseline/pilot sample IDs differ")
    base_by_id = {row["id"]: row for row in base_rows}
    pilot_rows = [pilot for pilot in pilot_rows if pilot["id"] in base_by_id]
    base_rows = [base_by_id[pilot["id"]] for pilot in pilot_rows]

    baseline = summarize(base_rows)
    pilot = summarize(pilot_rows)
    if abs(baseline["wer"] - 0.027297193387158785) > 1e-6:
        raise SystemExit(f"baseline WER mismatch: {baseline['wer']}")
    if abs(pilot["wer"] - pilot_accuracy["wer"]) > 1e-6:
        raise SystemExit("pilot eval WER does not match its per-utterance rows")

    # Paired bootstrap over utterances.
    rng = np.random.default_rng(42)
    base_errors, pilot_errors, words = [], [], []
    for b, p in zip(base_rows, pilot_rows):
        bw = jiwer.process_words([b["reference"]], [b["hypothesis"]])
        pw = jiwer.process_words([p["reference"]], [p["hypothesis"]])
        base_errors.append(bw.substitutions + bw.deletions + bw.insertions)
        pilot_errors.append(pw.substitutions + pw.deletions + pw.insertions)
        words.append(bw.hits + bw.substitutions + bw.deletions)
    base_errors = np.asarray(base_errors)
    pilot_errors = np.asarray(pilot_errors)
    words = np.asarray(words)
    n = len(words)

    def bootstrap(indices: np.ndarray) -> dict:
        local_base = base_errors[indices]
        local_pilot = pilot_errors[indices]
        local_words = words[indices]
        if len(local_words) == 0:
            return {"n_utterances": 0, "ci95_pp": [None, None],
                    "p_pilot_not_better": None}
        draws = []
        for _ in range(10000):
            pick = rng.integers(0, len(local_words), len(local_words))
            draws.append(
                (local_base[pick].sum() - local_pilot[pick].sum())
                / local_words[pick].sum() * 100
            )
        draws = np.asarray(draws)
        return {
            "n_utterances": int(len(local_words)),
            "delta_improvement_pp_mean": float(draws.mean()),
            "ci95_pp": [float(np.percentile(draws, 2.5)),
                         float(np.percentile(draws, 97.5))],
            "p_pilot_not_better": float((draws <= 0).mean()),
        }

    deltas = []
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        deltas.append(
            (base_errors[idx].sum() - pilot_errors[idx].sum()) / words[idx].sum() * 100
        )
    deltas = np.asarray(deltas)
    overall_delta = (baseline["wer"] - pilot["wer"]) * 100
    short_mask = np.asarray(
        [len(row["reference"].split()) <= SHORT_WORDS for row in base_rows]
    )
    bootstrap_slices = {
        "short": bootstrap(np.where(short_mask)[0]),
        "long": bootstrap(np.where(~short_mask)[0]),
    }
    short_base = slice_stats(base_rows, True)
    short_pilot = slice_stats(pilot_rows, True)
    long_base = slice_stats(base_rows, False)
    long_pilot = slice_stats(pilot_rows, False)
    long_delta = (long_pilot["wer"] - long_base["wer"]) * 100
    short_delta = (short_base["wer"] - short_pilot["wer"]) * 100

    size = json.loads(PILOT_SIZE.read_text())
    verify = json.loads(PILOT_VERIFY.read_text())
    pilot_result = json.loads(PILOT_RESULT.read_text())
    artifact_bytes = size["packed_artifact_bytes_measured"]
    criteria = {
        "dev256_wer_improvement_ge_0.10pp": overall_delta >= 0.10,
        "long_utterance_regression_le_0.10pp": long_delta <= LONG_REGRESSION_LIMIT_PP,
        "artifact_below_1GB": artifact_bytes < 1_000_000_000,
        "packed_reload_passed": bool(verify["passed"]),
    }
    payload = {
        "purpose": "paired dev-256 comparison of short-oversample pilot vs Strategy A",
        "protocol": {
            "split": "dev-clean",
            "n": len(base_rows),
            "seed": 42,
            "decoding": "greedy",
            "short_definition": f"reference word count <= {SHORT_WORDS}",
            "test_used": False,
        },
        "baseline": baseline,
        "pilot": pilot,
        "delta_pilot_minus_baseline": {
            "wer_pp": -overall_delta,
            "cer_pp": (pilot["cer"] - baseline["cer"]) * 100,
            "substitutions": pilot["substitutions"] - baseline["substitutions"],
            "deletions": pilot["deletions"] - baseline["deletions"],
            "insertions": pilot["insertions"] - baseline["insertions"],
            "word_errors": pilot["word_errors"] - baseline["word_errors"],
        },
        "slices": {
            "short": {"baseline": short_base, "pilot": short_pilot,
                      "delta_improvement_pp": short_delta},
            "long": {"baseline": long_base, "pilot": long_pilot,
                     "delta_pilot_minus_baseline_pp": long_delta},
        },
        "bootstrap": {
            "resamples": 10000,
            "seed": 42,
            "delta_improvement_pp_mean": float(deltas.mean()),
            "ci95_pp": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_pilot_not_better": float((deltas <= 0).mean()),
            "by_slice": bootstrap_slices,
        },
        "artifact": {
            "bytes": artifact_bytes,
            "scale_mode": size.get("scale_mode"),
            "n_bitlinear_layers": size.get("n_bitlinear_layers"),
            "packed_reload_passed": bool(verify["passed"]),
            "packed_reload_max_abs_logit_diff": verify.get("max_abs_logit_diff"),
            "packed_reload_argmax_agreement": verify.get("mean_argmax_agreement"),
        },
        "checkpoint": {
            "best_step": pilot_result.get("checkpoint", {}).get("best_step"),
            "selection": pilot_result.get("dev_selection"),
            "init_checkpoint": pilot_result.get("init_checkpoint"),
            "sampling": pilot_result.get("sampling"),
        },
        "criteria": criteria,
        "success": all(criteria.values()),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
