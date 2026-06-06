# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for unary elementwise kernels."""

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt

import pytest
import torch

from coreai_torch import (
    MetalParameter,
    TorchMetalKernel,
)

from ..utils import validate_numerical_output


@dataclass
class UnaryTest:
    """Parameter definition for unary elementwise tests."""

    operation: str
    inputs: list[float]
    results: list[float]
    torch_defn: Callable[[torch.Tensor], torch.Tensor]
    src_code: str | None = None
    decimal: int | None = None
    templates: dict[str, str] = field(default_factory=dict)


FUSED_KERNEL_SRC = """
// Guard against out-of-bounds threads
uint length = x.get_extent(0);
if (id >= length) return;

TYPE elt = x[id];
output[id] = sin(cos(sqrt(elt))) + elt;
"""


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests run only on Mac")
@pytest.mark.parametrize(
    "params",
    [
        UnaryTest(
            "cos",
            [i * pi for i in range(1, 5)],
            [-1, 1, -1, 1],
            torch.cos,
        ),
        UnaryTest("abs", list([-0.5, 0.5] * 5), [0.5] * 10, torch.abs),
        UnaryTest(
            "relu",
            list([-0.5, 0.5] * 5),
            [0.0, 0.5] * 5,
            torch.relu,
            src_code="output[id] = max(x[id], TYPE(0.0f));",
            templates={"x": "TYPE"},
        ),
        UnaryTest(
            "fused",
            [i * pi for i in range(10)],
            [sin(cos(sqrt(i * pi))) + (i * pi) for i in range(10)],
            lambda x: torch.sin(torch.cos(torch.sqrt(x))) + x,
            src_code=FUSED_KERNEL_SRC,
            decimal=0,
            templates={"x": "TYPE"},
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
async def test_elementwise_unary_metal_kernels(
    params: UnaryTest,
    dtype: torch.dtype,
) -> None:
    """We should be able to define and run unary elementwise kernels."""

    def torch_fn(x: torch.Tensor) -> torch.Tensor:
        return params.torch_defn(x)

    src = (
        params.src_code
        if params.src_code
        else f"output[id] = {params.operation}(x[id]);"
    )

    custom_kernel = TorchMetalKernel(
        f"custom_{params.operation}",
        input_names=["x"],
        result_names=["output"],
        src=src,
        torch_defn=torch_fn,
        metal_params=[
            MetalParameter("id", "uint", "thread_position_in_grid"),
        ],
        template_dtypes=params.templates,
    )

    class MetalModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return custom_kernel(
                x,
                threads_per_grid=(x.shape[0], 1, 1),
                threads_per_thread_group=(1, 1, 1),
                result_shapes=[list(x.shape)],
            )

    decimal = params.decimal if params.decimal is not None else 6
    await validate_numerical_output(
        model=MetalModel().eval(),
        custom_kernels=[custom_kernel],
        metal_inputs=True,
        input_names=["x"],
        output_names=["result"],
        atol=1.5 * 10**-decimal,
        rtol=0,
        x=torch.tensor(params.inputs, dtype=dtype),
    )


async def test_multi_result_kernel() -> None:
    """Users should be able to specify kernels with multiple returns."""

    def torch_fn(x: torch.Tensor) -> list[torch.Tensor]:
        return [torch.sin(x), torch.cos(x)]

    kernel = TorchMetalKernel(
        "multi_output",
        input_names=["x"],
        result_names=["output_sin", "output_cos"],
        src="output_sin[gid] = cos(x[gid]); output_cos[gid] = sin(x[gid]);",
        torch_defn=torch_fn,
        metal_params=[MetalParameter("gid", "uint", "thread_position_in_grid")],
    )

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            kernel_res = kernel(
                x,
                threads_per_grid=(x.shape[0], 1, 1),
                threads_per_thread_group=(1, 1, 1),
                result_shapes=[list(x.shape), list(x.shape)],
            )
            x_sin, x_cos = kernel_res[0], kernel_res[1]
            return x_sin + x_cos

    test_tn = torch.rand(25, dtype=torch.float32)
    await validate_numerical_output(
        model=Model().eval(),
        custom_kernels=[kernel],
        metal_inputs=True,
        input_names=["x"],
        output_names=["result"],
        x=test_tn,
    )
