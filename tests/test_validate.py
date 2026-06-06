# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for coreai_torch._validate — exported program validation."""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from coreai_torch import TorchConverter
from coreai_torch._validate import validate_exported_program


def test_non_decomposed_program_raises() -> None:
    """Passing a non-decomposed ExportedProgram raises a ValueError
    telling the user to call run_decompositions()."""

    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    model = SimpleModel().eval()
    x = torch.rand(2, 4)
    program = torch.export.export(model, args=(x,))

    with pytest.raises(ValueError, match="run_decompositions"):
        TorchConverter().add_exported_program(program)


def test_unsupported_core_aten_op_raises() -> None:
    """An unsupported core ATen op raises a ValueError mentioning
    register_torch_lowering()."""

    class SimpleModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.linalg.solve_triangular(x, x, upper=True)

    model = SimpleModel().eval()
    x = torch.rand(3, 3)
    program = torch.export.export(model, args=(x,))
    program = program.run_decompositions()

    with pytest.raises(ValueError, match="register_torch_lowering"):
        validate_exported_program(program, {})


def test_user_lowering_bypasses_unsupported_check() -> None:
    """Providing a user_lowerings entry for an unsupported op prevents the error."""

    class SimpleModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.linalg.solve_triangular(x, x, upper=True)

    model = SimpleModel().eval()
    x = torch.rand(3, 3)
    program = torch.export.export(model, args=(x,))
    program = program.run_decompositions()

    # Find the unsupported target string so we can build the right key.
    unsupported_targets = [
        node.target
        for node in program.graph.nodes
        if node.op == "call_function"
        and str(node.target).startswith("aten.linalg_solve_triangular")
    ]
    assert unsupported_targets, "Expected to find solve_triangular op in graph"

    target_str = str(unsupported_targets[0])
    # Build the qualified key: "aten::<op_name>"
    resolver_key = target_str[len("aten.") :]
    qualified_key = f"aten::{resolver_key}"

    # Providing a dummy lowering should suppress the error.
    validate_exported_program(program, {qualified_key: lambda *a: None})


def test_composite_ops_pass_validation() -> None:
    """Composite ops (e.g. scaled_dot_product_attention) should not be flagged."""

    class SimpleModel(nn.Module):
        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    model = SimpleModel().eval()
    q = k = v = torch.rand(1, 4, 8, 16)
    program = torch.export.export(model, args=(q, k, v))
    # Use default decompositions (sdpa stays because it's composite).
    program = program.run_decompositions()

    # Should not raise.
    validate_exported_program(program, {})


def test_validation_via_add_pytorch_module() -> None:
    """Validation also triggers through the add_pytorch_module() path."""

    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    model = SimpleModel().eval()
    x = torch.rand(2, 4)

    # export_fn intentionally skips run_decompositions().
    with pytest.raises(ValueError, match="run_decompositions"):
        TorchConverter().add_pytorch_module(
            model,
            export_fn=lambda m: torch.export.export(m, args=(x,)),
        )


def test_error_message_lists_ops() -> None:
    """The error message for non-decomposed programs lists the offending ops."""

    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    model = SimpleModel().eval()
    x = torch.rand(2, 4)
    program = torch.export.export(model, args=(x,))

    with pytest.raises(ValueError, match=r"aten\.linear"):
        TorchConverter().add_exported_program(program)
