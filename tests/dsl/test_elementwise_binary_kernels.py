# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for binary elementwise kernels."""

import sys
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from coreai_torch import (
    MetalParameter,
    TorchMetalKernel,
)

from ..utils import validate_numerical_output


@dataclass
class BinaryTest:
    """Parameter definition for binary elementwise tests."""

    operation: str
    inputs: tuple[list[float], list[float]]
    results: list[float]
    torch_defn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    src_code: str


def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Elementwise addition of two tensors."""
    return x + y


def torch_pow(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Elementwise pow of two tensors."""
    return torch.pow(x, y)


def torch_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Elementwise max of two tensors."""
    return torch.max(x, y)


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests run only on Mac")
@pytest.mark.parametrize(
    "params",
    [
        BinaryTest(
            "add",
            (list(range(10)), list(range(10, 20))),
            [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            torch_add,
            "result[id] = x[id] + y[id];",
        ),
        # BinaryTest(
        #     "max",
        #     (
        #         [1, 4, 5, 8],
        #         [2, 3, 6, 10],
        #     ),
        #     [2, 4, 6, 10],
        #     torch_max,
        #     "result[id] = max(x[id], y[id]);",
        # ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    # [torch.float16, torch.float32, torch.int8, torch.uint8],
    [torch.float32],
)
async def test_elementwise_binary_metal_kernels(
    params: BinaryTest,
    dtype: torch.dtype,
) -> None:
    """We should be able to define and run binary elementwise kernels."""
    await _run_binary_kernel_test(params, dtype)


@pytest.mark.parametrize(
    "params",
    [
        BinaryTest(
            "pow",
            (list(range(10)), [2] * 10),
            [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
            torch_pow,
            "result[id] = pow(x[id], y[id]);",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
async def test_elementwise_binary_float_kernels(
    params: BinaryTest,
    dtype: torch.dtype,
) -> None:
    """Tests for operations restricted only to floats."""
    await _run_binary_kernel_test(params, dtype)


async def _run_binary_kernel_test(
    params: BinaryTest,
    dtype: torch.dtype,
) -> None:
    """Runner for binary tests."""
    custom_kernel = TorchMetalKernel(
        f"custom_{params.operation}",
        input_names=["x", "y"],
        result_names=["result"],
        src=params.src_code,
        torch_defn=params.torch_defn,
        metal_params=[
            MetalParameter("id", "uint", "thread_position_in_grid"),
        ],
    )

    class MetalModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return custom_kernel(
                x,
                y,
                threads_per_grid=(x.shape[0], 1, 1),
                threads_per_thread_group=(1, 1, 1),
                result_shapes=[list(x.shape)],
            )

    x_tn = torch.tensor(params.inputs[0], dtype=dtype)
    y_tn = torch.tensor(params.inputs[1], dtype=dtype)

    await validate_numerical_output(
        model=MetalModel().eval(),
        custom_kernels=[custom_kernel],
        metal_inputs=True,
        input_names=["x", "y"],
        output_names=["result"],
        x=x_tn,
        y=y_tn,
    )
