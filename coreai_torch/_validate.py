# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Validation helpers for exported programs before conversion."""

from __future__ import annotations

import torch

from ._aten_to_core import _aten_to_core_resolver
from ._decomp import _COMPOSITE_OPS

# Cache the default decomposition table at module level to avoid repeated
# computation on every call.
_DEFAULT_DECOMPS: dict | None = None


def _get_default_decomps() -> dict:
    global _DEFAULT_DECOMPS
    if _DEFAULT_DECOMPS is None:
        _DEFAULT_DECOMPS = torch.export.default_decompositions()
    return _DEFAULT_DECOMPS


def validate_exported_program(
    ep: torch.export.ExportedProgram,
    user_lowerings: dict[str, object],
) -> None:
    """Validate that an exported program is ready for conversion.

    Raises ``ValueError`` with an actionable message when:

    1. The program contains ops that should have been decomposed by
       ``run_decompositions()`` — the caller forgot to call it.
    2. The program contains core ATen ops that are not supported by the
       converter — the user needs ``register_torch_lowering()``.
    """
    composite_targets = {str(op) for op in _COMPOSITE_OPS}
    default_decomps = _get_default_decomps()
    decomp_targets = {str(op) for op in default_decomps}

    # Assertion ops that preprocess_graph() strips before conversion.
    assertion_targets = {
        str(torch.ops.aten._assert_async.msg),
        str(torch.ops.aten._assert_scalar.default),
        str(torch.ops.aten.sym_constrain_range_for_size.default),
        str(torch.ops.aten.sym_constrain_range.default),
        str(torch.ops.aten._assert_tensor_metadata.default),
    }

    non_decomposed: list[str] = []
    unsupported: list[str] = []

    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue

        target = node.target
        target_str = str(target)

        # Skip composite ops — these are intentionally preserved.
        if target_str in composite_targets:
            continue

        # Skip assertion ops — preprocess_graph() removes them before conversion.
        if target_str in assertion_targets:
            continue

        # Check if the op should have been decomposed.
        # But first check if the resolver can handle it directly.
        if target_str in decomp_targets:
            parts = target_str.split("::", 1)
            resolver_key = parts[1] if len(parts) > 1 else target_str
            if target_str.startswith("aten."):
                resolver_key = target_str[len("aten.") :]
            qualified_target = f"aten::{resolver_key}"
            if (
                resolver_key not in _aten_to_core_resolver
                and qualified_target not in user_lowerings
            ):
                non_decomposed.append(target_str)
            continue

        # Only check aten ops for unsupported status.
        if not target_str.startswith("aten."):
            continue

        # Build the resolver key the same way converter.py does:
        # strip the namespace prefix and use the base target name.
        # e.g. "aten.foo.default" -> "foo.default"
        parts = target_str.split("::", 1)
        resolver_key = parts[1] if len(parts) > 1 else target_str
        # The FX node target string uses dots: "aten.foo.default"
        # but the resolver key format is "foo.default" (namespace stripped).
        if target_str.startswith("aten."):
            resolver_key = target_str[len("aten.") :]

        qualified_target = f"aten::{resolver_key}"

        if (
            resolver_key not in _aten_to_core_resolver
            and qualified_target not in user_lowerings
        ):
            unsupported.append(target_str)

    if non_decomposed:
        unique = sorted(set(non_decomposed))
        ops_list = ", ".join(unique)
        raise ValueError(
            f"The exported program contains non-decomposed ops: {ops_list}. "
            f"Please call run_decompositions() on your ExportedProgram before "
            f"passing it to TorchConverter. Example:\n"
            f"  ep = ep.run_decompositions(coreai_torch.get_decomp_table())"
        )

    if unsupported:
        unique = sorted(set(unsupported))
        ops_list = ", ".join(unique)
        raise ValueError(
            f"The exported program contains unsupported ATen ops: {ops_list}. "
            f"Use register_torch_lowering() to provide a custom lowering for "
            f"these ops."
        )
