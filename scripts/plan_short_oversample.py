"""Build the deterministic short-utterance oversampling plan for the pilot.

Only train.100 is read. The script records the original 10k subset distribution,
the selected 2k composition, exact row indices, effective sampling weights, and
the overlap check against the pinned dev-256 IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from onebit_asr.config import load_config
from onebit_asr.data.dataset import DATASET_ID, CONFIG
from onebit_asr.data.text import normalize_text

ROOT = Path(__file__).resolve().parent.parent
SHORT_WORDS = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def words(text: str) -> int:
    return len(normalize_text(text).split())


def duration(row: dict) -> float:
    raw = row["audio"].get("bytes")
    if raw is None:
        info = sf.info(row["audio"]["path"])
    else:
        info = sf.info(io.BytesIO(raw))
    return float(info.frames) / float(info.samplerate)


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": ordered[len(ordered) // 4],
        "median": ordered[len(ordered) // 2],
        "p75": ordered[3 * len(ordered) // 4],
        "max": ordered[-1],
        "mean": float(np.mean(ordered)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build short-utterance pilot plan.")
    parser.add_argument("--config", default="configs/attn_2bit_A_short_oversample.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/attn_2bit_on_2bitffn_A_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=10000)
    parser.add_argument("--subset-size", type=int, default=2000)
    parser.add_argument("--target-short-fraction", type=float, default=0.25)
    parser.add_argument("--output", default="results/short_oversample_plan.json")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    checkpoint = ROOT / args.checkpoint
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    actual_sha = sha256(checkpoint / "model.safetensors")
    if actual_sha != manifest["sha256"]["model.safetensors"]:
        raise SystemExit("A checkpoint hash does not match its manifest")

    ds = load_dataset(DATASET_ID, CONFIG, split="train.100")
    ds = ds.cast_column("audio", Audio(decode=False))
    id_to_index = {str(row["id"]): i for i, row in enumerate(ds.select_columns(["id"]))}
    shuffled = ds.shuffle(seed=args.seed)
    pool = shuffled.select(range(args.pool_size))
    pool_ids = [str(x) for x in pool["id"]]
    pool_indices = [id_to_index[x] for x in pool_ids]
    pool_texts = list(pool["text"])
    pool_wc = [words(t) for t in pool_texts]
    pool_dur = [duration(row) for row in pool]

    full_wc = [words(t) for t in ds["text"]]
    full_short = sum(1 for x in full_wc if x <= SHORT_WORDS)
    full_words = sum(full_wc)
    full_short_words = sum(x for x in full_wc if x <= SHORT_WORDS)

    short_idx = [i for i, x in enumerate(pool_wc) if x <= SHORT_WORDS]
    long_idx = [i for i, x in enumerate(pool_wc) if x > SHORT_WORDS]
    n_short = int(round(args.subset_size * args.target_short_fraction))
    n_long = args.subset_size - n_short
    if n_long > len(long_idx):
        raise SystemExit(f"need {n_long} long rows, pool has {len(long_idx)}")

    rng = np.random.default_rng(args.seed)
    if n_short > len(short_idx):
        short_selected = list(short_idx) + list(
            rng.choice(short_idx, n_short - len(short_idx), replace=True)
        )
    else:
        short_selected = list(rng.choice(short_idx, n_short, replace=False))
    long_selected = list(rng.choice(long_idx, n_long, replace=False))
    selected = short_selected + long_selected
    rng.shuffle(selected)

    samples = []
    for i in selected:
        samples.append({
            "index": int(pool_indices[i]),
            "id": pool_ids[i],
            "text": pool_texts[i],
            "n_words": int(pool_wc[i]),
            "duration_s": float(pool_dur[i]),
            "short": bool(pool_wc[i] <= SHORT_WORDS),
        })
    selected_words = sum(s["n_words"] for s in samples)
    short_words = sum(s["n_words"] for s in samples if s["short"])
    natural_short_frac = len(short_idx) / len(pool_wc)
    natural_short_word_share = (
        sum(x for x in pool_wc if x <= SHORT_WORDS) / sum(pool_wc)
    )
    target_short_frac = n_short / args.subset_size
    weights = {
        "short": target_short_frac / natural_short_frac,
        "long": (1.0 - target_short_frac) / (1.0 - natural_short_frac),
    }
    weights["short_to_long_ratio"] = weights["short"] / weights["long"]

    dev_payload = json.loads(
        (ROOT / "results/ctc_decoding_A_full_dev256.json").read_text()
    )
    dev_rows = dev_payload["per_utterance"]
    dev_ids = {row["id"] for row in dev_rows}
    selected_ids = {s["id"] for s in samples}
    overlap = sorted(selected_ids & dev_ids)
    if overlap:
        raise SystemExit(f"train/dev ID overlap: {overlap[:5]}")

    short_dur = [s["duration_s"] for s in samples if s["short"]]
    long_dur = [s["duration_s"] for s in samples if not s["short"]]
    plan = {
        "purpose": "deterministic 500-step short-utterance oversampling pilot plan",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": actual_sha,
        "checkpoint_selection": manifest.get("selection"),
        "dataset": f"{DATASET_ID}/{CONFIG}/train.100",
        "seed": args.seed,
        "short_definition": f"normalized reference word count <= {SHORT_WORDS}",
        "pool_size": args.pool_size,
        "subset_size": args.subset_size,
        "target_short_fraction": target_short_frac,
        "n_short": n_short,
        "n_long": n_long,
        "unique_short_rows": len(short_idx),
        "unique_long_rows": len(long_idx),
        "short_repeat_count": n_short - len(short_idx) if n_short > len(short_idx) else 0,
        "short_repeat_factor": n_short / len(short_idx),
        "natural_short_fraction": natural_short_frac,
        "natural_short_word_share": natural_short_word_share,
        "target_short_word_share": short_words / selected_words,
        "effective_sampling_weights": weights,
        "sampling_frequency": {"short": target_short_frac, "long": 1 - target_short_frac},
        "distribution": {
            "train_100_words": distribution(full_wc),
            "train_100_short_count": full_short,
            "train_100_short_fraction": full_short / len(full_wc),
            "train_100_short_word_share": full_short_words / full_words,
            "original_10k_words": distribution(pool_wc),
            "original_10k_duration_s": distribution(pool_dur),
            "original_10k_short_count": len(short_idx),
            "original_10k_short_fraction": natural_short_frac,
            "original_10k_short_word_share": natural_short_word_share,
            "selected_short_duration_s": distribution(short_dur),
            "selected_long_duration_s": distribution(long_dur),
            "dev_256_short_count": sum(1 for r in dev_rows if r["ref_words"] <= SHORT_WORDS),
            "dev_256_short_fraction": sum(1 for r in dev_rows if r["ref_words"] <= SHORT_WORDS) / len(dev_rows),
            "dev_256_short_word_share": sum(r["ref_words"] for r in dev_rows if r["ref_words"] <= SHORT_WORDS) / sum(r["ref_words"] for r in dev_rows),
            "dev_256_short_error_share": sum(r["S"] + r["D"] + r["I"] for r in dev_rows if r["ref_words"] <= SHORT_WORDS) / sum(r["S"] + r["D"] + r["I"] for r in dev_rows),
        },
        "leakage_check": {"dev_256_overlap": overlap, "test_used": False, "split": "train.100"},
        "samples": samples,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: plan[k] for k in (
        "checkpoint_sha256", "short_definition", "pool_size", "subset_size",
        "target_short_fraction", "n_short", "n_long", "unique_short_rows",
        "short_repeat_count", "natural_short_fraction", "natural_short_word_share",
        "target_short_word_share", "effective_sampling_weights", "distribution",
        "leakage_check")}, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
