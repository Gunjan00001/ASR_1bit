"""Read-only PBS resource preflight for the full FFN comparison.

This script does not submit anything. It answers:

    Can a 4-CPU / 1-GPU / 64-GB / 4-hour job start now, and are the full-run
    output paths still unused?

It reads ``qstat``, ``pbsnodes``, ``nvidia-smi`` (over SSH), and the local disk,
then writes a JSON report. The full-run job is submitted only when ``ready`` is
true; a busy node is reported, never bypassed or retried automatically.

Debugging notes
---------------
- PBS reports a node as ``job-busy`` when jobs occupy it; the authoritative
  schedulable quantity is ``resources_available.ncpus - resources_assigned.ncpus``.
  A free GPU alone is not enough.
- ``qstat -u`` truncates long job names (e.g. ``qat_ffn_f*``), so duplicate-job
  detection compares the first eight characters and strips the ``*`` marker.
- Memory values from ``pbsnodes`` use mixed units (``kb``/``mb``); parse them to
  bytes before comparing.
- The GPU query is read-only and uses the cluster's existing SSH path. If SSH
  fails, the report marks GPU availability unknown and is not ready.
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL_OUTPUTS = (
    "results/qat_ffn_control_full.json",
    "results/qat_ffn_mixed_full.json",
)
FULL_CHECKPOINT_DIRS = (
    "checkpoints/qat_ffn_control_full",
    "checkpoints/qat_ffn_mixed_full",
)
_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmgt]?b?)$", re.IGNORECASE)


def parse_size(value: str) -> int:
    """PBS memory string (``263029300kb``, ``2062226mb``, ``0b``) -> bytes."""
    text = str(value).strip()
    match = _SIZE_RE.fullmatch(text)
    if not match:
        raise ValueError(f"unrecognized size: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    factors = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}
    return int(number * factors[unit])


def parse_pbsnodes(text: str, node: str) -> dict:
    """Extract the ``key = value`` fields for one PBS node block."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == node), None)
    if start is None:
        return {}
    fields: dict[str, str] = {}
    for line in lines[start + 1:]:
        if not line.strip():
            break
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def parse_qstat_jobs(text: str) -> list[str]:
    """Job names from the short ``qstat -u`` table, preserving truncation."""
    names = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and re.match(r"^\d+\.", parts[0]):
            names.append(parts[3].rstrip("*"))
    return names


def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse ``index, memory.free, utilization.gpu`` CSV rows (units MB)."""
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "free_mb": int(parts[1]),
                "utilization_pct": int(parts[2]),
            })
        except ValueError:
            continue
    return gpus


def _run(cmd: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def decide_ready(
    *,
    node_state: str,
    free_cpus: int,
    free_host_mem_bytes: int,
    gpu_free_count: int,
    disk_free_bytes: int,
    required_cpus: int,
    required_host_mem_bytes: int,
    required_gpus: int,
    required_disk_bytes: int,
    duplicate_jobs: list[str],
    gpu_query_error: str | None,
) -> list[str]:
    """Return blocking reasons; empty means a job may be submitted."""
    reasons = []
    if node_state == "offline":
        reasons.append("PBS node is offline")
    if free_cpus < required_cpus:
        reasons.append(f"free CPUs {free_cpus} < required {required_cpus}")
    if free_host_mem_bytes < required_host_mem_bytes:
        reasons.append(
            f"free host memory {free_host_mem_bytes / 1e9:.1f} GB < "
            f"required {required_host_mem_bytes / 1e9:.0f} GB"
        )
    if gpu_query_error:
        reasons.append(f"GPU query failed: {gpu_query_error}")
    if gpu_free_count < required_gpus:
        reasons.append(f"free GPUs {gpu_free_count} < required {required_gpus}")
    if disk_free_bytes < required_disk_bytes:
        reasons.append(
            f"free disk {disk_free_bytes / 1e9:.1f} GB < "
            f"required {required_disk_bytes / 1e9:.0f} GB"
        )
    if duplicate_jobs:
        reasons.append(f"duplicate jobs already queued/running: {duplicate_jobs}")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only full-run PBS preflight.")
    parser.add_argument("--node", default="jiit-gpu01")
    parser.add_argument("--queue", default="gpu")
    parser.add_argument("--job-name", default="qat_ffn_full")
    parser.add_argument("--user", default=getpass.getuser())
    parser.add_argument("--required-cpus", type=int, default=2)
    parser.add_argument("--required-gpus", type=int, default=1)
    parser.add_argument("--required-gpu-free-gb", type=float, default=25.0)
    parser.add_argument("--required-host-free-gb", type=float, default=64.0)
    parser.add_argument("--required-disk-gb", type=float, default=40.0)
    parser.add_argument("--output", default=str(ROOT / "results" / "full_run_preflight.json"))
    args = parser.parse_args()

    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        pass

    qstat = _run(["qstat", "-u", args.user])
    jobs = parse_qstat_jobs(qstat.stdout) if qstat.returncode == 0 else []
    prefix = args.job_name[:8]
    duplicate_jobs = [name for name in jobs if name.startswith(prefix)]

    pbs = _run(["pbsnodes", "-a"])
    fields = parse_pbsnodes(pbs.stdout, args.node) if pbs.returncode == 0 else {}
    available_cpus = int(fields.get("resources_available.ncpus", 0) or 0)
    assigned_cpus = int(fields.get("resources_assigned.ncpus", 0) or 0)
    free_cpus = max(0, available_cpus - assigned_cpus)
    try:
        available_mem = parse_size(fields.get("resources_available.mem", "0b"))
        assigned_mem = parse_size(fields.get("resources_assigned.mem", "0b"))
    except ValueError:
        available_mem = assigned_mem = 0
    free_host_mem = max(0, available_mem - assigned_mem)

    gpu_query_error = None
    gpus: list[dict] = []
    try:
        remote = _run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", args.node,
            "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], timeout=15)
        if remote.returncode != 0:
            gpu_query_error = (remote.stderr or remote.stdout or "ssh/nvidia-smi failed").strip()
        else:
            gpus = parse_nvidia_smi(remote.stdout)
    except Exception as exc:  # noqa: BLE001
        gpu_query_error = f"{type(exc).__name__}: {exc}"

    gpu_free_count = sum(
        1 for gpu in gpus if gpu["free_mb"] >= args.required_gpu_free_gb * 1024
    )
    disk_free = shutil.disk_usage(ROOT).free

    existing_outputs = [p for p in FULL_OUTPUTS if (ROOT / p).exists()]
    partial_checkpoints = []
    for path in FULL_CHECKPOINT_DIRS:
        root = ROOT / path
        if root.is_dir() and any(p.name.startswith("checkpoint-") for p in root.iterdir()):
            partial_checkpoints.append(path)

    required_host = int(args.required_host_free_gb * 1024**3)
    required_disk = int(args.required_disk_gb * 1024**3)
    reasons = decide_ready(
        node_state=fields.get("state", "unknown"),
        free_cpus=free_cpus,
        free_host_mem_bytes=free_host_mem,
        gpu_free_count=gpu_free_count,
        disk_free_bytes=disk_free,
        required_cpus=args.required_cpus,
        required_host_mem_bytes=required_host,
        required_gpus=args.required_gpus,
        required_disk_bytes=required_disk,
        duplicate_jobs=duplicate_jobs,
        gpu_query_error=gpu_query_error,
    )
    ready = not reasons

    report = {
        "purpose": "read-only preflight for the full FFN control vs mixed-precision run",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "node": args.node,
        "queue": args.queue,
        "job_name": args.job_name,
        "pbs": {
            "state": fields.get("state"),
            "available_cpus": available_cpus,
            "assigned_cpus": assigned_cpus,
            "free_cpus": free_cpus,
            "available_host_mem_bytes": available_mem,
            "assigned_host_mem_bytes": assigned_mem,
            "free_host_mem_bytes": free_host_mem,
            "resources_available_ngpus": fields.get("resources_available.ngpus"),
        },
        "gpus": gpus,
        "gpu_free_count_above_threshold": gpu_free_count,
        "gpu_query_error": gpu_query_error,
        "disk_free_bytes": disk_free,
        "existing_jobs": jobs,
        "duplicate_jobs": duplicate_jobs,
        "existing_outputs": existing_outputs,
        "partial_checkpoints": partial_checkpoints,
        "requirements": {
            "cpus": args.required_cpus,
            "gpus": args.required_gpus,
            "gpu_free_gb": args.required_gpu_free_gb,
            "host_free_gb": args.required_host_free_gb,
            "disk_free_gb": args.required_disk_gb,
        },
        "ready": ready,
        "blocked_reasons": reasons,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"Node: {args.node} state={report['pbs']['state']} "
          f"free CPUs={free_cpus}/{available_cpus}")
    print(f"GPUs: {len(gpus)} visible; {gpu_free_count} with "
          f"≥{args.required_gpu_free_gb:.0f} GB free")
    print(f"Host free memory: {free_host_mem / 1e9:.1f} GB; "
          f"disk free: {disk_free / 1e9:.1f} GB")
    print(f"Duplicate jobs: {duplicate_jobs or 'none'}")
    print(f"Ready: {ready}")
    for reason in reasons:
        print(f"  BLOCKED: {reason}")
    print(f"Report: {out}")
    return 0 if ready else 2


if __name__ == "__main__":
    sys.exit(main())
