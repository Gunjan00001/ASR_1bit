"""CTC decoding utilities: greedy, blank-penalized greedy, and prefix beam search."""

from __future__ import annotations

import math

import numpy as np
import torch


def ctc_collapse(token_ids: torch.Tensor | list[int], blank_id: int = 0) -> list[int]:
    """Remove repeated labels and blanks from a frame-level CTC path."""
    if torch.is_tensor(token_ids):
        token_ids = token_ids.detach().cpu().tolist()
    out: list[int] = []
    previous = None
    for token in token_ids:
        token = int(token)
        if token != blank_id and token != previous:
            out.append(token)
        previous = token
    return out


def greedy_decode(
    logits: torch.Tensor,
    blank_id: int = 0,
    blank_penalty: float = 0.0,
) -> list[int]:
    """Argmax-frame CTC decoding with an optional additive blank penalty.

    ``blank_penalty > 0`` makes blank less likely; the same logit adjustment is
    used by :func:`prefix_beam_search`.
    """
    if logits.dim() != 2:
        raise ValueError(f"expected logits [frames, vocab], got {tuple(logits.shape)}")
    adjusted = logits.detach().float().clone()
    if blank_penalty:
        adjusted[:, blank_id] -= float(blank_penalty)
    return ctc_collapse(adjusted.argmax(dim=-1), blank_id=blank_id)


def prefix_beam_search(
    logits: torch.Tensor,
    beam_width: int = 10,
    blank_id: int = 0,
    blank_penalty: float = 0.0,
    nbest: int = 1,
) -> list[tuple[float, list[int]]]:
    """CTC prefix beam search without an external language model.

    Returns up to ``nbest`` ``(log_probability, token_ids)`` hypotheses, best
    first. The blank penalty is applied to the blank logit before log-softmax,
    matching :func:`greedy_decode`.
    """
    if logits.dim() != 2:
        raise ValueError(f"expected logits [frames, vocab], got {tuple(logits.shape)}")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    adjusted = logits.detach().float().clone()
    if blank_penalty:
        adjusted[:, blank_id] -= float(blank_penalty)
    logp = torch.log_softmax(adjusted, dim=-1).cpu().numpy()
    n_tokens = logp.shape[1]

    beams: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, -math.inf)
    }
    for frame in logp:
        next_beams: dict[tuple[int, ...], list[float]] = {}

        def add(prefix: tuple[int, ...], blank: bool, value: float) -> None:
            if not math.isfinite(value):
                return
            bucket = next_beams.setdefault(prefix, [-math.inf, -math.inf])
            bucket[0 if blank else 1] = float(
                np.logaddexp(bucket[0 if blank else 1], value)
            )

        for prefix, (p_blank, p_nonblank) in beams.items():
            p_total = float(np.logaddexp(p_blank, p_nonblank))
            add(prefix, True, p_total + float(frame[blank_id]))
            last = prefix[-1] if prefix else None
            for token in range(n_tokens):
                if token == blank_id:
                    continue
                token_p = float(frame[token])
                if token == last:
                    add(prefix, False, p_nonblank + token_p)
                    add(prefix + (token,), False, p_blank + token_p)
                else:
                    add(prefix + (token,), False, p_total + token_p)

        ranked = sorted(
            next_beams.items(),
            key=lambda item: float(np.logaddexp(item[1][0], item[1][1])),
            reverse=True,
        )[:beam_width]
        beams = {prefix: (values[0], values[1]) for prefix, values in ranked}

    final = sorted(
        ((float(np.logaddexp(pb, pnb)), list(prefix)) for prefix, (pb, pnb) in beams.items()),
        key=lambda item: item[0],
        reverse=True,
    )
    return final[:nbest]
