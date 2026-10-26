"""Training callbacks for validation-based checkpoint selection.

What this does
--------------
- Evaluates a fixed dev subset every ``eval_every_steps`` optimizer steps.
- Evaluates the **final** state at ``on_train_end`` even when the final step is
  not on the cadence. This matters for 10k/1,875-step runs: the last checkpoint
  must be considered, not silently discarded in favor of step 1,500.
- Keeps the best state on CPU and restores it before the script's final
  evaluation/checkpoint.
- Optionally persists that best state and history to ``state_path`` atomically,
  so a Trainer resume after an interruption can continue the same selection.
- Injects ``eval_fn`` so the callback is testable without a model/dataset/GPU.

Debugging notes
---------------
- The returned summary distinguishes ``final_scheduled_dev_wer`` (last cadence
  eval) from ``final_state_dev_wer`` (the actual final training step). The
  selection uses both plus the earlier history.
- The best-state file is large (full state dict); save it only when the best
  improves and once at train end. The temporary file + ``os.replace`` prevents a
  partial write from corrupting a previous best.
- A corrupt state file raises rather than silently starting a fresh selection;
  that would make a resumed run's checkpoint selection irreproducible.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

_STATE_VERSION = 1


class DevWERCallback(TrainerCallback):
    """Evaluate dev WER, keep the best state, and optionally persist it."""

    def __init__(
        self,
        eval_fn,
        *,
        eval_every_steps: int = 500,
        verbose: bool = True,
        state_path: str | Path | None = None,
        load_state: bool = True,
    ) -> None:
        if eval_every_steps <= 0:
            raise ValueError("eval_every_steps must be positive")
        self.eval_fn = eval_fn
        self.eval_every_steps = int(eval_every_steps)
        self.verbose = verbose
        self.state_path = Path(state_path) if state_path is not None else None

        self.history: list[dict] = []
        self.best_wer: float = math.inf
        self.best_step: int | None = None
        self.best_state: dict | None = None
        self.final_wer: float | None = None
        self.final_scheduled_wer: float | None = None
        self.final_state_wer: float | None = None
        self.final_state_cer: float | None = None
        self.final_state_step: int | None = None
        self.restored_best: bool = False
        self.resumed: bool = False
        self.model = None
        self._last_eval_step: int | None = None

        if load_state and self.state_path is not None and self.state_path.is_file():
            self._load_state()

    # --- persistence -------------------------------------------------------
    def _load_state(self) -> None:
        payload = torch.load(self.state_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("version") != _STATE_VERSION:
            raise ValueError(f"unrecognized dev-selection state file: {self.state_path}")
        self.history = list(payload.get("history", []))
        self.best_wer = float(payload.get("best_wer", math.inf))
        self.best_step = payload.get("best_step")
        self.best_state = payload.get("best_state")
        self.final_wer = payload.get("final_wer")
        self.final_scheduled_wer = payload.get("final_scheduled_wer")
        self.final_state_wer = payload.get("final_state_wer")
        self.final_state_cer = payload.get("final_state_cer")
        self.final_state_step = payload.get("final_state_step")
        if self.history:
            self._last_eval_step = int(self.history[-1].get("step", 0)) or None
        self.resumed = True
        if self.verbose:
            print(
                f"  [dev] resumed selection state from {self.state_path} "
                f"({len(self.history)} evals; best WER={self.best_wer:.4f} "
                f"at step {self.best_step})",
                flush=True,
            )

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _STATE_VERSION,
            "history": list(self.history),
            "best_wer": self.best_wer,
            "best_step": self.best_step,
            "best_state": self.best_state,
            "final_wer": self.final_wer,
            "final_scheduled_wer": self.final_scheduled_wer,
            "final_state_wer": self.final_state_wer,
            "final_state_cer": self.final_state_cer,
            "final_state_step": self.final_state_step,
        }
        tmp = Path(str(self.state_path) + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, self.state_path)

    # --- evaluation --------------------------------------------------------
    def _evaluate(self, model, step: int, phase: str) -> dict:
        t0 = time.perf_counter()
        metrics = self.eval_fn(model)
        return {
            "step": int(step),
            "wer": float(metrics["wer"]),
            "cer": float(metrics["cer"]),
            "duration_s": time.perf_counter() - t0,
            "phase": phase,
        }

    def _consider(self, record: dict, model) -> None:
        self.history.append(record)
        self.final_wer = record["wer"]
        improved = record["wer"] < self.best_wer
        if improved:
            self.best_wer = record["wer"]
            self.best_step = record["step"]
            # CPU clone: training continues on the GPU, so a view/GPU reference
            # would be overwritten or moved.
            self.best_state = {
                key: value.detach().to("cpu").clone()
                for key, value in model.state_dict().items()
            }
        if self.verbose:
            print(
                f"  [dev] {record['phase']} step {record['step']}: "
                f"WER={record['wer']:.4f} CER={record['cer']:.4f} "
                f"({record['duration_s']:.1f}s)",
                flush=True,
            )
        if improved:
            self._save_state()

    # --- Trainer events ----------------------------------------------------
    def on_train_begin(self, args, state, control, model=None, **kwargs):  # noqa: D102
        # On a resumed run, the Trainer starts at the checkpoint step without
        # replaying the step where that checkpoint was written. Evaluate that
        # state if the persisted history does not already contain it, so the
        # selection is not forced to skip a checkpoint across the interruption.
        if state.global_step <= 0:
            return
        if self._last_eval_step == state.global_step:
            return
        if model is None:
            model = self.model
        if model is None:
            return
        self.model = model
        record = self._evaluate(model, int(state.global_step), "resume_state")
        self._last_eval_step = int(state.global_step)
        self._consider(record, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: D102
        # `global_step` counts optimizer steps, not micro-batches. Eval only at
        # the configured cadence and never at step 0. Guard duplicate calls for
        # the same step (Trainer versions differ in callback timing).
        if state.global_step <= 0 or state.global_step % self.eval_every_steps != 0:
            return
        if self._last_eval_step == state.global_step:
            return
        if model is None:
            model = self.model
        if model is None:
            return
        self.model = model
        self._last_eval_step = int(state.global_step)
        record = self._evaluate(model, int(state.global_step), "scheduled")
        self.final_scheduled_wer = record["wer"]
        self._consider(record, model)

    def on_train_end(self, args, state, control, **kwargs):  # noqa: D102
        model = kwargs.get("model", self.model)
        if model is None:
            return
        self.model = model
        final_step = int(state.global_step)
        if final_step > 0:
            existing = next(
                (r for r in reversed(self.history) if int(r["step"]) == final_step),
                None,
            )
            if existing is None:
                # The final step (e.g. 1,875) may not be on the 500-step cadence.
                # Evaluate it before restoring the best so it is a real candidate.
                record = self._evaluate(model, final_step, "final_state")
                self.final_state_wer = record["wer"]
                self.final_state_cer = record["cer"]
                self.final_state_step = final_step
                self._consider(record, model)
                self._last_eval_step = final_step
            else:
                self.final_state_wer = float(existing["wer"])
                self.final_state_cer = float(existing["cer"])
                self.final_state_step = final_step
            self._save_state()

        if self.best_state is not None:
            device = next(model.parameters()).device
            model.load_state_dict(
                {key: value.to(device) for key, value in self.best_state.items()}
            )
            self.restored_best = True
            if self.verbose:
                print(
                    f"  [dev] restored best state from step {self.best_step} "
                    f"(WER={self.best_wer:.4f})",
                    flush=True,
                )

    def summary(self) -> dict:
        """Serializable summary for the result JSON."""
        return {
            "eval_every_steps": self.eval_every_steps,
            "history": list(self.history),
            "best_dev_wer": self.best_wer if self.history else None,
            "best_step": self.best_step,
            # final_dev_wer is the finally evaluated state (final step if it was
            # not already on the cadence); the scheduled value is kept separately.
            "final_dev_wer": self.final_wer,
            "final_scheduled_dev_wer": self.final_scheduled_wer,
            "final_state_dev_wer": self.final_state_wer,
            "final_state_cer": self.final_state_cer,
            "final_state_step": self.final_state_step,
            "restored_best": self.restored_best,
            "resumed": self.resumed,
            "state_path": str(self.state_path) if self.state_path is not None else None,
        }
