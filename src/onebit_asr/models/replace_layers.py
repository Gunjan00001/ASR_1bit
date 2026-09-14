"""Configurable ``nn.Linear`` -> ``BitLinear`` replacement (Phase 8).

Selection is by taxonomy category (``onebit_asr.models.conformer_utils``), not
by hard-coded architecture assumptions. Weights are copied numerically from
the FP layer into the BitLinear master weights *before* quantization, so the
pre-quantization values are preserved and only ``forward`` binarizes.
"""

from __future__ import annotations

from torch import nn

from onebit_asr.models.conformer_utils import (
    categorize,
    get_parent_module,
    iter_linears,
    select_by_category,
)
from onebit_asr.quantization.bitlinear import BitLinear


def _describe(name: str, module: nn.Linear, category: str) -> dict:
    return {
        "name": name,
        "category": category,
        "in_features": module.in_features,
        "out_features": module.out_features,
        "params": sum(p.numel() for p in module.parameters()),
    }


def replace_linears(
    model: nn.Module,
    selection: dict[str, bool],
    layer_cls: type[nn.Linear] = BitLinear,
    scale_mode: str = "per_tensor_mean_abs",
    ste_clip: float | None = None,
) -> dict:
    """Replace selected ``nn.Linear`` layers in-place with ``layer_cls``.

    Args:
        model: model to modify in place.
        selection: category -> bool mapping (validated against the taxonomy; an
            unknown key raises ``ValueError`` instead of silently matching
            nothing).
        layer_cls: replacement class; must provide ``from_linear`` (BitLinear
            does). Defaults to :class:`BitLinear`.
        scale_mode / ste_clip: quantization configuration passed to the
            replacement layer.

    Returns:
        A report dict with ``replaced`` / ``retained`` lists, counts, and the
        parameter count of each group.
    """
    selected_names = {name for name, _m, _c in select_by_category(model, selection)}

    replaced, retained = [], []
    # Materialize first: we mutate the module tree during the loop.
    for name, module in list(iter_linears(model)):
        category = categorize(name)
        if name in selected_names:
            if isinstance(module, layer_cls):
                replaced.append({**_describe(name, module, category), "note": "already replaced"})
                continue
            parent, attr = get_parent_module(model, name)
            replacement = layer_cls.from_linear(module, scale_mode=scale_mode, ste_clip=ste_clip)
            setattr(parent, attr, replacement)
            replaced.append(_describe(name, replacement, category))
        else:
            retained.append(_describe(name, module, category))

    return {
        "replaced": replaced,
        "retained": retained,
        "n_replaced": len(replaced),
        "n_retained": len(retained),
        "replaced_params": sum(r["params"] for r in replaced),
        "retained_params": sum(r["params"] for r in retained),
        "scale_mode": scale_mode,
        "ste_clip": ste_clip,
        "layer_cls": layer_cls.__name__,
    }


def summarize_by_category(entries: list[dict]) -> dict:
    """Group report entries by category with counts and parameter totals."""
    out: dict = {}
    for entry in entries:
        bucket = out.setdefault(entry["category"], {"count": 0, "params": 0})
        bucket["count"] += 1
        bucket["params"] += entry["params"]
    return out


def report_replacement(report: dict) -> str:
    """Render a replacement report as a human-readable string."""
    lines = [
        f"Replacement: {report['layer_cls']}  "
        f"(scale_mode={report['scale_mode']}, ste_clip={report['ste_clip']})",
        f"  replaced: {report['n_replaced']} layers, {report['replaced_params']:,} params",
        f"  retained: {report['n_retained']} layers, {report['retained_params']:,} params",
    ]
    for label in ("replaced", "retained"):
        by_cat = summarize_by_category(report[label])
        if not by_cat:
            continue
        lines.append(f"  {label} by category:")
        for cat, stats in sorted(by_cat.items(), key=lambda kv: -kv[1]["params"]):
            lines.append(f"    {cat:<20} {stats['count']:>4} layers  {stats['params']:>14,} params")
    return "\n".join(lines)
