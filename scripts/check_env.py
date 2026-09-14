"""Phase 0 — environment diagnostic.

Imports every required dependency and reports versions + hardware.
Exit code 0 means the environment satisfies Phase 0 acceptance.
"""

import importlib
import json
import platform
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

REQUIRED = [
    "torch",
    "torchaudio",
    "transformers",
    "datasets",
    "evaluate",
    "jiwer",
    "accelerate",
    "numpy",
    "soundfile",
    "yaml",
]


DIST_NAMES = {"yaml": "pyyaml"}  # module -> distribution name


def dist_version(name: str) -> str:
    from importlib.metadata import version

    try:
        return version(DIST_NAMES.get(name, name))
    except Exception:  # noqa: BLE001
        return "unknown"


def try_import(name):
    try:
        importlib.import_module(name)
        return {"ok": True, "version": dist_version(name)}
    except Exception as exc:  # noqa: BLE001 - diagnostic must not crash
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: try_import(name) for name in REQUIRED},
    }

    if report["packages"]["torch"]["ok"]:
        import torch

        report["torch_cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_version"] = torch.version.cuda or "n/a (CPU build)"
        report["torch_cpu_threads"] = torch.get_num_threads()
    else:
        report["torch_cuda_available"] = False

    ok = all(info["ok"] for info in report["packages"].values())

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "env.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print("\nPhase 0 acceptance:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
