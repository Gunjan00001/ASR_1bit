import json
from pathlib import Path

from transformers import AutoConfig, Wav2Vec2Processor

from onebit_asr.serving.manager import ModelManager, ServingError

ROOT = Path(__file__).resolve().parent.parent


def test_frontend_assets_are_present():
    for name in ("index.html", "styles.css", "app.js", "audio-worklet.js"):
        path = ROOT / "frontend" / name
        assert path.is_file(), name
        assert path.stat().st_size > 100
    html = (ROOT / "frontend/index.html").read_text()
    assert "/static/app.js" in html
    assert "Start recording" in html
    assert "chunked-overlap" in html.lower() or "streaming" in html.lower()


def test_deploy_packages_have_config_and_tokenizer():
    for model_id in ("ffn2bit", "strategy_a", "fp32"):
        path = ROOT / "deploy/models" / model_id
        assert (path / "metadata.json").is_file()
        assert (path / "config.json").is_file()
        assert (path / "vocab.json").is_file()
        config = AutoConfig.from_pretrained(path)
        processor = Wav2Vec2Processor.from_pretrained(path)
        assert config.pad_token_id == 0
        assert processor.tokenizer.pad_token_id == 0
        meta = json.loads((path / "metadata.json").read_text())
        assert meta["id"] == model_id
        assert meta["full_test_wer"] is not None
        assert meta["artifact_bytes"] > 0


def test_sha256_checksums_are_recorded():
    for model_id in ("ffn2bit", "strategy_a", "fp32"):
        lines = (ROOT / "deploy/models" / model_id / "SHA256SUMS").read_text().splitlines()
        assert lines
        assert any(len(line.split()[0]) == 64 for line in lines)


def test_lfs_pattern_is_configured():
    attrs = (ROOT / ".gitattributes").read_text()
    assert "deploy/models/**/*.safetensors" in attrs
    assert "filter=lfs" in attrs


def test_model_manager_metadata_and_unknown_model():
    manager = ModelManager(device="cpu")
    ids = [m["id"] for m in manager.available_models()]
    assert ids[:3] == ["ffn2bit", "strategy_a", "fp32"]
    meta = manager.public_metadata("strategy_a")
    assert meta["artifact_bytes"] == 989716588
    assert meta["full_test_wer"] == 0.027369902617163724
    assert meta["artifact_available"] is True
    try:
        manager.public_metadata("not-a-model")
    except ServingError:
        pass
    else:
        raise AssertionError("unknown model should raise")
