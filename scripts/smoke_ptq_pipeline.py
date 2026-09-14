"""One-off smoke check: replacement + BitLinear forward + size accounting.

Runs a 1-utterance forward through the binarized checkpoint (no evaluation
protocol involved) and exercises binary_model_size_report. Fast (<1 min
besides model load); the full 256-utterance PTQ run is scripts/quantize.py.
"""

from pathlib import Path

from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.evaluation.model_size import binary_model_size_report
from onebit_asr.models.replace_layers import replace_linears, report_replacement

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg = load_config(ROOT / "configs" / "binary.yaml")
    model_id = cfg["model"]

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
    model.eval()

    report = replace_linears(
        model, cfg["layers"], scale_mode=cfg["quantization"]["scale"],
        ste_clip=cfg["quantization"].get("ste_clip"),
    )
    print(report_replacement(report))

    sample = load_dev_subset(subset_size=256, seed=cfg["eval"]["seed"])[0]
    inputs = processor(sample["audio"], sampling_rate=16_000, return_tensors="pt")
    logits = model(inputs.input_values).logits
    print("logits shape:", tuple(logits.shape))
    assert logits.shape[0] == 1 and logits.shape[-1] == 32, logits.shape

    size = binary_model_size_report(model, repo_id=model_id, scale_mode=cfg["quantization"]["scale"])
    for key, value in size.items():
        print(f"  {key}: {value}")
    print("SMOKE CHECK PASSED")


if __name__ == "__main__":
    main()
