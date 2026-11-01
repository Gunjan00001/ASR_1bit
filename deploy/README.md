# Deployment notes

The live demo is launched with:

```bash
.venv/bin/python scripts/run_streaming_app.py --host 127.0.0.1 --port 7860 --device auto
```

The model package directories are in `models/`. Large packed weights are LFS
tracked, not ordinary Git blobs. If LFS is unavailable, local development uses
the verified v2 artifacts under `checkpoints/` as a fallback.

GitHub free-tier LFS capacity is 1 GB storage / 1 GB monthly bandwidth. The two
selected artifacts total ~2.12 GB, so a paid data pack or an explicitly
documented release-asset fallback is required for both. Do not commit partial
weights.
