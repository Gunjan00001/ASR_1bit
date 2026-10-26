"""Checkpoint-resume helpers for the QAT Trainer.

The stock HF ``Trainer`` writes ``checkpoint-<step>/`` directories when
``save_strategy="steps"``. The full-run job needs to find the newest one
deterministically after an interruption. Keeping this logic in one tiny module
makes it unit-testable without loading a 593M-parameter model.

Debugging notes
---------------
- Only directories named exactly ``checkpoint-<integer>`` with a
  ``trainer_state.json`` are valid. A stray directory or the final
  ``save_qat_checkpoint`` output (``model.safetensors`` directly under the run
  directory) must not be mistaken for a resumable Trainer checkpoint.
- ``max()`` on the parsed integer step, not lexicographic sorting (``checkpoint-1000``
  sorts before ``checkpoint-500`` as a string).
"""

from __future__ import annotations

import re
from pathlib import Path

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(root: str | Path, *, require_trainer_state: bool = True) -> Path | None:
    """Return the highest-step valid Trainer checkpoint under ``root``.

    Args:
        root: run directory containing ``checkpoint-*`` subdirectories.
        require_trainer_state: require ``trainer_state.json`` so a half-written
            directory is not resumed. Keep the default for real resumes.

    Returns:
        The checkpoint directory, or ``None`` if none exists.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = _CHECKPOINT_RE.fullmatch(child.name)
        if not match or not child.is_dir():
            continue
        if require_trainer_state and not (child / "trainer_state.json").is_file():
            continue
        candidates.append((int(match.group(1)), child))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def list_checkpoints(root: str | Path) -> list[Path]:
    """Return valid Trainer checkpoints sorted by step ascending."""
    root = Path(root)
    if not root.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = _CHECKPOINT_RE.fullmatch(child.name)
        if match and child.is_dir() and (child / "trainer_state.json").is_file():
            found.append((int(match.group(1)), child))
    return [path for _step, path in sorted(found, key=lambda item: item[0])]
