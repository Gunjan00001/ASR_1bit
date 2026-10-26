"""Unit tests for the read-only PBS preflight parsers and decision logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "check_cluster_resources", ROOT / "scripts" / "check_cluster_resources.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ccr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ccr)


def test_parse_size_units():
    assert ccr.parse_size("0b") == 0
    assert ccr.parse_size("1kb") == 1024
    assert ccr.parse_size("2062226mb") == 2062226 * 1024**2
    assert ccr.parse_size("671088640kb") == 671088640 * 1024


def test_parse_pbsnodes_extracts_free_cpus_and_memory():
    text = """jiit-gpu01
     Mom = jiit-gpu01
     state = job-busy
     resources_available.ncpus = 96
     resources_assigned.ncpus = 94
     resources_available.mem = 2062226mb
     resources_assigned.mem = 671088640kb
     resources_available.ngpus = 8

jiit-cn01
     state = free
"""
    fields = ccr.parse_pbsnodes(text, "jiit-gpu01")
    assert fields["state"] == "job-busy"
    free_cpus = int(fields["resources_available.ncpus"]) - int(
        fields["resources_assigned.ncpus"]
    )
    assert free_cpus == 2
    assert fields["resources_available.ngpus"] == "8"
    assert ccr.parse_pbsnodes(text, "missing-node") == {}


def test_parse_qstat_jobs_handles_truncated_names():
    text = """Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
12772.jiit-master* jiit_99* gpu qat_ffn_f* 727428 1 4 64gb 02:30 R 00:00
"""
    assert ccr.parse_qstat_jobs(text) == ["qat_ffn_f"]


def test_parse_nvidia_smi_rows():
    text = "0, 78000, 0\n1, 12000, 90\ngarbage\n"
    rows = ccr.parse_nvidia_smi(text)
    assert rows == [
        {"index": 0, "free_mb": 78000, "utilization_pct": 0},
        {"index": 1, "free_mb": 12000, "utilization_pct": 90},
    ]


def test_decide_ready_reports_cpu_shortfall_and_duplicate():
    reasons = ccr.decide_ready(
        node_state="job-busy",
        free_cpus=2,
        free_host_mem_bytes=1_000 * 1024**3,
        gpu_free_count=8,
        disk_free_bytes=100 * 1024**3,
        required_cpus=4,
        required_host_mem_bytes=64 * 1024**3,
        required_gpus=1,
        required_disk_bytes=40 * 1024**3,
        duplicate_jobs=[],
        gpu_query_error=None,
    )
    assert any("free CPUs" in reason for reason in reasons)

    ready = ccr.decide_ready(
        node_state="free",
        free_cpus=2,
        free_host_mem_bytes=1_000 * 1024**3,
        gpu_free_count=8,
        disk_free_bytes=100 * 1024**3,
        required_cpus=2,
        required_host_mem_bytes=64 * 1024**3,
        required_gpus=1,
        required_disk_bytes=40 * 1024**3,
        duplicate_jobs=[],
        gpu_query_error=None,
    )
    assert ready == []

    duplicate = ccr.decide_ready(
        node_state="free",
        free_cpus=2,
        free_host_mem_bytes=1_000 * 1024**3,
        gpu_free_count=8,
        disk_free_bytes=100 * 1024**3,
        required_cpus=2,
        required_host_mem_bytes=64 * 1024**3,
        required_gpus=1,
        required_disk_bytes=40 * 1024**3,
        duplicate_jobs=["qat_ffn_f"],
        gpu_query_error=None,
    )
    assert any("duplicate" in reason for reason in duplicate)
