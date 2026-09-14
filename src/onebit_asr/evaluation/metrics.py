"""WER/CER computation (inputs must already be normalized)."""

import jiwer


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    return float(jiwer.wer(references, hypotheses))


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    return float(jiwer.cer(references, hypotheses))
