"""Launch the live streaming ASR research demo."""

from __future__ import annotations

import argparse
import os

import uvicorn

from onebit_asr.serving.api import create_app
from onebit_asr.serving.manager import ModelManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Live chunked-streaming ASR demo.")
    parser.add_argument("--host", default=os.environ.get("ASR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ASR_PORT", "7860")))
    parser.add_argument("--device", default=os.environ.get("ASR_DEVICE", "auto"),
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    manager = ModelManager(device=args.device)
    app = create_app(manager)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
