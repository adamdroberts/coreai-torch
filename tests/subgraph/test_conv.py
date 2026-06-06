# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Validate that Conv op sub-graph patterns are correctly imported and lowered."""

import pytest
import torch
from torch import nn
from torch.export import Dim

from ..utils import validate_numerical_output


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("f16", [True, False])
@pytest.mark.parametrize("dynamic", [True, False])
async def test_conv2d_with_batch_norm(bias: bool, f16: bool, dynamic: bool) -> None:
    """Test conv2D + batch norm."""

    class Model(nn.Module):
        """Validate Conv2D with bias."""

        def __init__(self, bias: bool):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=3,
                out_channels=5,
                kernel_size=[3, 3],
                stride=[2, 2],
                bias=bias,
            )
            self.bn = torch.nn.BatchNorm2d(num_features=5, eps=0.1, momentum=0.1)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Invoke conv2d."""
            return self.bn(self.conv2d(input))

    model = Model(bias=bias)
    if f16:
        model = model.half()
    model_inp = torch.randn(
        (1, 3, 14, 14),
        dtype=torch.float16 if f16 else torch.float32,
    )
    _ = model(model_inp)
    dynamic_shapes = None
    if dynamic:
        dynamic_shapes = {"input": {2: Dim("height", min=14), 3: Dim("width", min=14)}}
    await validate_numerical_output(
        model=model,
        input=model_inp,
        dynamic_shapes=dynamic_shapes,
    )


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("f16", [True, False])
@pytest.mark.parametrize("dynamic", [True, False])
async def test_conv3d_with_batch_norm(bias: bool, f16: bool, dynamic: bool) -> None:
    """Test conv3D + batch norm."""

    class Model(nn.Module):
        """Validate Conv3D with bias."""

        def __init__(self, bias: bool):
            super().__init__()
            self.conv2d = torch.nn.Conv3d(
                in_channels=3,
                out_channels=5,
                kernel_size=[3, 3, 3],
                stride=[2, 2, 2],
                bias=bias,
            )
            self.bn = torch.nn.BatchNorm3d(num_features=5, eps=0.1, momentum=0.1)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Invoke conv2d."""
            return self.bn(self.conv2d(input))

    model = Model(bias=bias)
    if f16:
        model = model.half()
    model_inp = torch.randn(
        (1, 3, 14, 14, 15),
        dtype=torch.float16 if f16 else torch.float32,
    )
    _ = model(model_inp)
    dynamic_shapes = None
    if dynamic:
        dynamic_shapes = {
            "input": {
                2: Dim("depth", min=14),
                3: Dim("height", min=14),
                4: Dim("width", min=15),
            },
        }
    await validate_numerical_output(
        model=model,
        input=model_inp,
        dynamic_shapes=dynamic_shapes,
    )
