"""Shared Linear-layer taxonomy and selection for Wav2Vec2-Conformer.

This is the SINGLE source of truth for how Linear layers are categorized.
It is used by:
- ``scripts/inspect_model.py`` (Phase 1 inspection)
- ``src/onebit_asr/models/replace_layers.py`` (Phase 8 replacement)

Keeping inspection and replacement on one taxonomy prevents the failure mode
where a config key silently matches nothing (e.g. selecting ``ffn`` when the
taxonomy emits ``ffn_in`` / ``ffn_out``).
"""

from __future__ import annotations

from torch import nn

#: Every category the taxonomy can produce.
CATEGORIES = (
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_output",
    "attention_other",
    "ffn_in",
    "ffn_out",
    "feature_projection",
    "ctc_head",
    "other",
)

#: Categories that are meaningful to select for quantization in this repo.
SELECTABLE_CATEGORIES = (
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_output",
    "ffn_in",
    "ffn_out",
    "feature_projection",
    "ctc_head",
    "other",
)


def categorize(name: str) -> str:
    """Map a fully-qualified module name to a taxonomy category.

    Handles both Hugging Face Wav2Vec2-Conformer naming (discovered in Phase 1:
    ``self_attn.linear_q``, ``ffn1.intermediate_dense``, ``ffn1.output_dense``,
    ``feature_projection.projection``, ``lm_head``) and fairseq-style naming
    (``q_proj``, ``linear1``, ``linear2``) for portability.
    """
    n = name.lower()

    if "lm_head" in n or n.endswith("ctc_head") or "ctc_head" in n:
        return "ctc_head"
    if "feature_projection" in n:
        return "feature_projection"

    if "attention" in n or "self_attn" in n or "attn" in n:
        if "q_proj" in n or "linear_q" in n or n.endswith(".query") or ".query." in n:
            return "attention_q"
        if "k_proj" in n or "linear_k" in n or n.endswith(".key") or ".key." in n:
            return "attention_k"
        if "v_proj" in n or "linear_v" in n or n.endswith(".value") or ".value." in n:
            return "attention_v"
        if "out_proj" in n or "linear_out" in n or "output" in n:
            return "attention_output"
        return "attention_other"

    if "linear1" in n or "intermediate_dense" in n or "intermediate" in n:
        return "ffn_in"
    if "linear2" in n or "output_dense" in n:
        return "ffn_out"
    return "other"


def iter_linears(model: nn.Module):
    """Yield ``(name, module)`` for every ``nn.Linear`` in ``model``.

    ``BitLinear`` subclasses ``nn.Linear``, so already-replaced layers are
    included; callers that need only pristine layers should check the type.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def selected_categories(selection: dict[str, bool]) -> set[str]:
    """Validate a category->bool selection mapping and return the True keys.

    Raises ``ValueError`` for unknown keys so a typo cannot silently exempt
    every layer it was meant to quantize.
    """
    unknown = sorted(set(selection) - set(CATEGORIES))
    if unknown:
        raise ValueError(
            f"Unknown layer-selection key(s): {unknown}. "
            f"Valid keys: {list(CATEGORIES)}"
        )
    return {k for k, v in selection.items() if v}


def select_by_category(model: nn.Module, selection: dict[str, bool]):
    """Return ``[(name, module, category)]`` for categories selected as True."""
    wanted = selected_categories(selection)
    out = []
    for name, module in iter_linears(model):
        cat = categorize(name)
        if cat in wanted:
            out.append((name, module, cat))
    return out


def get_parent_module(model: nn.Module, qualified_name: str):
    """Return ``(parent_module, attribute_name)`` for a dotted module name."""
    if "." not in qualified_name:
        return model, qualified_name
    parent_path, attr = qualified_name.rsplit(".", 1)
    return model.get_submodule(parent_path), attr
