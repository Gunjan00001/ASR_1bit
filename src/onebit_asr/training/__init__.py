"""Training: FP baseline train loop, QAT assembly (Phase 11), distillation."""

from onebit_asr.training.qat import (
    CHECKPOINT_FORMAT,
    bitlinear_grad_stats,
    build_training_arguments,
    build_trainer,
    freeze_feature_extractor,
    freeze_report,
    load_qat_checkpoint,
    make_loss_recorder,
    master_weight_delta,
    master_weight_snapshot,
    probe_gradient_flow,
    save_qat_checkpoint,
    setup_qat_model,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "setup_qat_model",
    "freeze_feature_extractor",
    "freeze_report",
    "bitlinear_grad_stats",
    "probe_gradient_flow",
    "master_weight_snapshot",
    "master_weight_delta",
    "build_training_arguments",
    "build_trainer",
    "make_loss_recorder",
    "save_qat_checkpoint",
    "load_qat_checkpoint",
]
