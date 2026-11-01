"""Build deployable model directories for the selected ASR models.

Copies small config/tokenizer/manifest files and writes metadata/checksums.
Weights are *not* copied by default; pass --copy-weights when preparing a Git
LFS upload. The local demo resolves the verified v2 artifacts in checkpoints/ if
a deploy weight file is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF = Path.home() / (
    ".cache/huggingface/hub/models--facebook--wav2vec2-conformer-rope-large-960h-ft/"
    "snapshots/6b36ef01c6443c67ae7ed0822876d091ab50e4aa"
)
DEPLOY = ROOT / "deploy/models"

MODELS = {
    "ffn2bit": {
        "display_name": "2-bit FFN",
        "kind": "packed",
        "checkpoint": "checkpoints/qat_ffn_mixed_2bit_full",
        "checkpoint_sha256": "87005f36bca859fe80bdb48f7a02a18841fed0a5e3495b2d877d2fd1a42befdb",
        "artifact_v2": "checkpoints/qat_ffn_mixed_2bit_full_packed_v2/model_packed.safetensors",
        "artifact_historical": "checkpoints/qat_ffn_mixed_2bit_full_packed/model_packed.safetensors",
        "artifact_bytes": 1131269460,
        "historical_artifact_bytes": 1131062436,
        "artifact_sha256": "e8e138050f2252a3b9aa327aec87cbe3340f244509ab2b6953d491bd8881c856",
        "full_test_wer": 0.025829275715155205,
        "dev_wer": 0.024605920799692427,
        "dev_wer_note": "packed v2 dev-256; QAT reference 0.024413687043444827",
        "quantization": {
            "bits": 2,
            "scale": "per_tensor_mean_abs",
            "attention_quantized": 0,
            "attention_fp32": 96,
            "ffn_quantized": 79,
            "ffn_fp32": 17,
            "bitlinear_layers": 79,
        },
        "description": "Accuracy winner. 2-bit FFN with the fixed 17 FP32 FFN exclusions and all attention FP32.",
        "limitations": [
            "Packed inference expands weights to FP32; not accelerated.",
            "Greedy CTC, no language model.",
            "Clean LibriSpeech only.",
        ],
    },
    "strategy_a": {
        "display_name": "Strategy A (sub-1 GB)",
        "kind": "packed",
        "checkpoint": "checkpoints/attn_2bit_on_2bitffn_A_full",
        "checkpoint_sha256": "fc9fb3dd3521809fe2d90b7150167f6a403dd72f3f69429142bea80e74b64358",
        "artifact_v2": "checkpoints/attn_2bit_on_2bitffn_A_full_packed_v2/model_packed.safetensors",
        "artifact_historical": "checkpoints/attn_2bit_on_2bitffn_A_full_packed/model_packed.safetensors",
        "artifact_bytes": 989716588,
        "historical_artifact_bytes": 989509572,
        "artifact_sha256": "182c62d18ef00af897a21506848f1ec8246070596c8611988294bc7f80293ca9",
        "full_test_wer": 0.027369902617163724,
        "dev_wer": 0.027297193387158785,
        "dev_wer_note": "packed v2 dev-256; matches QAT",
        "quantization": {
            "bits": 2,
            "scale": "per_tensor_mean_abs",
            "attention_quantized": 36,
            "attention_fp32": 60,
            "ffn_quantized": 79,
            "ffn_fp32": 17,
            "bitlinear_layers": 115,
        },
        "description": "Compression winner. 36 attention projections and 79 FFN layers 2-bit; 60 attention + 17 FFN projections FP32; below 1 GB.",
        "limitations": [
            "Packed inference expands weights to FP32; not accelerated.",
            "Slightly higher WER than 2-bit FFN.",
            "Clean LibriSpeech only.",
        ],
    },
    "fp32": {
        "display_name": "FP32 reference",
        "kind": "fp32",
        "model_id": "facebook/wav2vec2-conformer-rope-large-960h-ft",
        "checkpoint_sha256": "97bb9761fb71ec1225100bc81ccf7d002e0d0ba3d0604c1fd2dbda7d7d491f1d",
        "artifact_bytes": 2373507712,
        "full_test_wer": 0.019267346317711504,
        "dev_wer": 0.020184544405997693,
        "quantization": None,
        "description": "Unquantized reference. Not uploaded; downloaded from Hugging Face at runtime if permitted.",
        "limitations": ["Requires the external Hugging Face model.", "Largest artifact."],
    },
}

TOKENIZER_FILES = [
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-weights", action="store_true",
                        help="copy v2 artifacts into deploy/models (for Git LFS)")
    args = parser.parse_args()
    DEPLOY.mkdir(parents=True, exist_ok=True)
    for model_id, spec in MODELS.items():
        out = DEPLOY / model_id
        out.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "packed":
            checkpoint = ROOT / spec["checkpoint"]
            shutil.copy2(checkpoint / "config.json", out / "config.json")
            artifact = ROOT / spec["artifact_v2"]
            shutil.copy2(
                Path(str(artifact) + ".manifest.json"),
                out / "model_packed.safetensors.manifest.json",
            )
            if args.copy_weights:
                shutil.copy2(artifact, out / "model_packed.safetensors")
        else:
            shutil.copy2(HF / "config.json", out / "config.json")
        for name in TOKENIZER_FILES:
            shutil.copy2(HF / name, out / name)
        metadata = {
            "id": model_id,
            **{k: v for k, v in spec.items() if k not in ("checkpoint", "artifact_v2", "artifact_historical")},
            "checkpoint": spec.get("checkpoint"),
            "artifact_v2": spec.get("artifact_v2"),
            "artifact_historical": spec.get("artifact_historical"),
            "deploy_artifact": (
                f"deploy/models/{model_id}/model_packed.safetensors"
                if spec["kind"] == "packed" else None
            ),
            "local_fallback_artifact": spec.get("artifact_v2"),
            "runtime": "FP32 expansion before F.linear" if spec["kind"] == "packed" else "FP32",
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
        readme = [
            f"# {spec['display_name']}", "",
            spec["description"], "",
            f"- Full test-clean WER: **{spec['full_test_wer']*100:.2f}%**",
            f"- Dev-256 WER: `{spec['dev_wer']}`",
            f"- Artifact: `{spec['artifact_bytes']:,} B`",
            f"- SHA-256: `{spec.get('artifact_sha256', spec.get('checkpoint_sha256'))}`",
            "", "## Runtime", "",
            "This model is served through `onebit_asr.serving.ModelManager`.",
            "Packed weights are expanded to FP32 before matrix multiplication;",
            "the artifact is a storage format, not an inference accelerator.",
        ]
        if spec["kind"] == "packed":
            readme += [
                "", "The deploy weight file is tracked with Git LFS in the hosted",
                "repository. Local development falls back to the verified v2",
                f"artifact at `{spec['artifact_v2']}`.",
            ]
        (out / "README.md").write_text("\n".join(readme) + "\n")
        checks = []
        for file in sorted(out.iterdir()):
            if file.name == "SHA256SUMS":
                continue
            if file.is_file():
                checks.append(f"{sha256(file)}  {file.name}")
        if spec["kind"] == "packed":
            checks.append(f"{spec['artifact_sha256']}  model_packed.safetensors")
        else:
            checks.append(f"{spec['checkpoint_sha256']}  model.safetensors")
        (out / "SHA256SUMS").write_text("\n".join(checks) + "\n")
        print(f"packaged {model_id} -> {out}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
