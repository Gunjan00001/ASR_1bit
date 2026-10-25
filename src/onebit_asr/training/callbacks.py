"""Training callbacks for validation-based checkpoint selection.

Why this exists
---------------
The historical QAT runs evaluated only after the final step and saved the final
model. That cannot tell whether an earlier epoch was better, and it makes a
quantizer comparison vulnerable to "the last step happened to be worse". This
callback evaluates a fixed dev subset during training, keeps the best state on
CPU, and restores it before the final evaluation/checkpoint.

Debugging notes
---------------
- The optimizer zeroes gradients; this callback never inspects them. It only
  runs a forward-only evaluation.
- `eval_fn` is injected (a closure over the pinned evaluator and dev samples).
  It must return a dict with at least `wer` and `cer`. This makes the callback
  unit-testable without a real model or dataset.
- The best state copy is a CPU clone of the full state dict. For the 593M model
  that is ~2.4 GB; only one copy is kept. Release it after saving the final
  checkpoint if memory is tight.
- `on_train_end` restores the best state; the training script must use the model
  after `trainer.train()` (not a separately reloaded final checkpoint) to get the
  selected weights.
"""

from __future__ import annotations

import math

from transformers import TrainerCallback


class DevWERCallback(TrainerCallback):
    """Evaluate dev WER every N optimizer steps and restore the best state."""

    def __init__(
        self,
        eval_fn,
        *,
        eval_every_steps: int = 500,
        verbose: bool = True,
    ) -> None:
        if eval_every_steps <= 0:
            raise ValueError("eval_every_steps must be positive")
        self.eval_fn = eval_fn
        self.eval_every_steps = int(eval_every_steps)
        self.verbose = verbose

        self.history: list[dict] = []
        self.best_wer: float = math.inf
        self.best_step: int | None = None
        self.best_state: dict | None = None
        self.final_wer: float | None = None
        self.restored_best: bool = False
        self.model = None
        self._last_eval_step: int | None = None

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: D102
        # `global_step` counts optimizer steps, not micro-batches. Eval only at
        # the configured cadence and never at step 0. Guard against duplicate
        # calls for the same step (Trainer versions differ in callback timing).
        if state.global_step <= 0 or state.global_step % self.eval_every_steps != 0:
            return
        if self._last_eval_step == state.global_step:
            return
        self._last_eval_step = int(state.global_step)
        if model is None:
            model = self.model
        if model is None:
            return
        self.model = model

        metrics = self.eval_fn(model)
        record = {
            "step": int(state.global_step),
            "wer": float(metrics["wer"]),
            "cer": float(metrics["cer"]),
        }
        self.history.append(record)
        self.final_wer = record["wer"]
        if self.verbose:
            print(
                f"  [dev] step {record['step']}: WER={record['wer']:.4f} "
                f"CER={record['cer']:.4f}",
                flush=True,
            )

        if record["wer"] < self.best_wer:
            self.best_wer = record["wer"]
            self.best_step = record["step"]
            # CPU clone: the model will continue training on the GPU, so a view
            # or a GPU reference would be overwritten/moved.
            self.best_state = {
                key: value.detach().to("cpu").clone()
                for key, value in model.state_dict().items()
            }

    def on_train_end(self, args, state, control, **kwargs):  # noqa: D102
        model = kwargs.get("model", self.model)
        if self.best_state is not None and model is not None:
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
            "final_dev_wer": self.final_wer,
            "restored_best": self.restored_best,
        }
