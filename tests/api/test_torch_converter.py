# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Tests that mirror the code examples in docs/api/conversion/TorchConverter.md.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from coreai._compiler.dialects import coreai

from coreai_torch import TorchConverter
from coreai_torch._utils import get_operand, get_operands

from ..utils import filecheck_pattern

# ---------------------------------------------------------------------------
# Custom op for register_torch_lowering examples — registered once at module level
# ---------------------------------------------------------------------------


@torch.library.custom_op("ti_test_lib::scaled_add", mutates_args=())
def scaled_add(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    return x + scale * y


@scaled_add.register_fake
def _(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export_resnet_like():
    """Small conv model that exercises adaptive_avg_pool2d."""

    class TinyResNetLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(8, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = self.pool(x)
            x = x.flatten(1)
            return self.fc(x)

    model = TinyResNetLike().eval()
    exported = torch.export.export(model, args=(torch.randn(1, 3, 8, 8),))
    return exported.run_decompositions()


def _export_scaled_add():
    class ScaledAddModel(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.ops.ti_test_lib.scaled_add(x, y, 0.5)

    model = ScaledAddModel().eval()
    exported = torch.export.export(model, args=(torch.randn(4, 8), torch.randn(4, 8)))
    return exported.run_decompositions()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTorchConverterDocs:
    """Validates the code examples in docs/api/conversion/TorchConverter.md.

    Each test corresponds to a specific constructor or method example so that
    documentation examples are kept honest by the test suite.
    """

    # --- Factory methods ----------------------------------------------------

    def test_constructor_example(self):
        """TorchConverter().add_exported_program(exported) succeeds."""
        model = nn.Linear(10, 5).eval()
        exported = torch.export.export(model, args=(torch.randn(1, 10),))
        exported = exported.run_decompositions()
        converter = TorchConverter().add_exported_program(exported)
        assert converter is not None

    # --- to_coreai --------------------------------------------------

    @pytest.mark.ir
    def test_to_coreai_with_names(self):
        """to_coreai example: input_names/output_names are reflected in the IR."""
        model = nn.Linear(10, 5).eval()
        exported = torch.export.export(model, args=(torch.randn(1, 10),))
        exported = exported.run_decompositions()

        converter = TorchConverter().add_exported_program(
            exported,
            input_names=["image"],
            output_names=["logits"],
        )
        coreai_program = converter.to_coreai()
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: coreai.name = "image"
                // CHECK: coreai.name = "logits"
            """,
        )

    # --- register_torch_lowering: override built-in ATen lowering -----------------

    def test_register_torch_lowering_override_adaptive_avg_pool2d(self):
        """register_torch_lowering example 1: static-shape override of adaptive_avg_pool2d."""
        exported = _export_resnet_like()
        converter = TorchConverter()

        @converter.register_torch_lowering(
            "aten::_adaptive_avg_pool2d.default", allow_override=True
        )
        def lower_adaptive_avg_pool2d_static(values_map, node, loc):
            x = get_operand(values_map, node, 0, loc)
            output_h, output_w = node.args[1]
            input_h, input_w = x.type.shape[2], x.type.shape[3]
            stride_h, stride_w = input_h // output_h, input_w // output_w
            kernel_h = input_h - (output_h - 1) * stride_h
            kernel_w = input_w - (output_w - 1) * stride_w
            return coreai.broadcasting_divide(
                coreai.sumpool2d(
                    x,
                    kernel_size=np.array([kernel_h, kernel_w], dtype=np.uint32),
                    strides=np.array([stride_h, stride_w], dtype=np.uint32),
                    dilation=coreai.constant([1, 1], dtype=np.uint32),
                ),
                coreai.cast(float(kernel_h * kernel_w), x.type.element_type),
            )

        coreai_program = converter.add_exported_program(exported).to_coreai()
        assert coreai_program is not None

    # --- register_torch_lowering: custom op lowering ------------------------------

    @pytest.mark.ir
    def test_register_torch_lowering_custom_op(self):
        """register_torch_lowering example 2: custom op lowering for ti_test_lib::scaled_add."""
        exported = _export_scaled_add()
        converter = TorchConverter()

        @converter.register_torch_lowering("ti_test_lib::scaled_add.default")
        def lower_scaled_add(values_map, node, loc):
            x, y = get_operands(values_map, node, [0, 1], loc)
            scale = node.args[2]
            scale_val = coreai.constant(scale, dtype=x.type.element_type)
            scaled_y = coreai.broadcasting_mul(y, scale_val, loc=loc)
            return coreai.broadcasting_add(x, scaled_y, loc=loc)

        coreai_program = converter.add_exported_program(
            exported,
            input_names=["x", "y"],
            output_names=["result"],
        ).to_coreai()
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: coreai.decomposable.broadcasting_mul
                // CHECK: coreai.decomposable.broadcasting_add
                // CHECK-NOT: ti_test_lib
            """,
        )
