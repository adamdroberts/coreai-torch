# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for matmul kernels."""

import sys
from secrets import choice

import pytest
import torch

from coreai_torch import (
    MetalParameter,
    TorchMetalKernel,
)

from ..utils import validate_numerical_output


def torch_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Perform matrix multiplication on two input tensors."""
    return torch.matmul(x, y)


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests run only on Mac")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.int8,
        torch.int16,
        torch.int32,
    ],
)
@pytest.mark.parametrize("dynamic", [True, False])
async def test_naive_matmul_kernel(
    dtype: torch.dtype,
    dynamic: bool,
    int_naive_matmul: str,
    float_naive_matmul: str,
    bfloat_naive_matmul: str,
) -> None:
    """We should be able to run a naive implementation of matmul."""
    if dtype in {torch.float16, torch.float32}:
        src = float_naive_matmul
    elif dtype is torch.bfloat16:
        src = bfloat_naive_matmul
    else:
        src = int_naive_matmul

    custom_matmul = TorchMetalKernel(
        "custom_matmul",
        input_names=["A", "B"],
        result_names=["C"],
        src=src,
        torch_defn=torch_matmul,
        metal_params=[
            MetalParameter("gid", "uint2", "thread_position_in_grid"),
        ],
        template_dtypes={"A": "TYPE"},
    )

    class MatmulModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 2, (
                f"expected x to be of rank 2, got rank {x.dim()} instead"
            )
            assert y.dim() == 2, (
                f"expected y to be of rank 2, got rank {x.dim()} instead"
            )
            M, K_x = x.shape  # noqa: N806
            K_y, N = y.shape  # noqa: N806
            assert K_x == K_y, f"expected equal dimensions, got {K_x} vs {K_y}"
            grid = (N, M, 1)
            threads = (16, 16, 1)
            result_shape = [M, N]
            return custom_matmul(
                x,
                y,
                threads_per_grid=grid,
                threads_per_thread_group=threads,
                result_shapes=[result_shape],
            )

    model = MatmulModel().eval()
    rng = range(2, 20)
    M, K, N = choice(rng), choice(rng), choice(rng)  # noqa: N806
    fuzzed_x = (
        torch.rand(M, K, dtype=dtype)
        if dtype in {torch.bfloat16, torch.float16, torch.float32}
        else torch.randint(100, (M, K), dtype=dtype)
    )
    fuzzed_y = (
        torch.rand(K, N, dtype=dtype)
        if dtype in {torch.bfloat16, torch.float16, torch.float32}
        else torch.randint(100, (K, N), dtype=dtype)
    )

    if dynamic:
        k_dim = torch.export.Dim(name="K", min=1, max=32)
        dynamic_shapes = {"x": {1: k_dim}, "y": {0: k_dim}}
    else:
        dynamic_shapes = None

    await validate_numerical_output(
        model=model,
        custom_kernels=[custom_matmul],
        metal_inputs=True,
        input_names=["x", "y"],
        output_names=["result"],
        atol=0.1,
        rtol=0.1,
        dynamic_shapes=dynamic_shapes,
        x=fuzzed_x,
        y=fuzzed_y,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.int8,
        torch.int16,
        torch.int32,
    ],
)
@pytest.mark.parametrize("dynamic", [True, False])
async def test_tiled_matmul_kernel(
    dtype: torch.dtype,
    dynamic: bool,
    int_tiled_matmul: str,
    float_tiled_matmul: str,
    bfloat_tiled_matmul: str,
) -> None:
    """We should be able to run a tiled implementation of matmul."""
    if dtype in {torch.float16, torch.float32}:
        src = float_tiled_matmul
    elif dtype is torch.bfloat16:
        src = bfloat_tiled_matmul
    else:
        src = int_tiled_matmul

    custom_matmul = TorchMetalKernel(
        "tiled_matmul",
        input_names=["A", "B"],
        result_names=["C"],
        src=src,
        torch_defn=torch_matmul,
        metal_params=[
            MetalParameter("gid", "uint2", "thread_position_in_grid"),
            MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
            MetalParameter("tgid", "uint2", "threadgroup_position_in_grid"),
        ],
        template_dtypes={"A": "TYPE"},
    )

    class MatmulModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 2, (
                f"expected x to be of rank 2, got rank {x.dim()} instead"
            )
            assert y.dim() == 2, (
                f"expected y to be of rank 2, got rank {x.dim()} instead"
            )
            M, K_x = x.shape  # noqa: N806
            K_y, N = y.shape  # noqa: N806
            assert K_x == K_y, f"expected equal dimensions, got {K_x} vs {K_y}"
            TILE = 16  # noqa: N806
            grid = (
                ((N + TILE - 1) // TILE) * TILE,
                ((M + TILE - 1) // TILE) * TILE,
                1,
            )
            threads = (TILE, TILE, 1)
            result_shape = [M, N]
            return custom_matmul(
                x,
                y,
                threads_per_grid=grid,
                threads_per_thread_group=threads,
                result_shapes=[result_shape],
            )

    model = MatmulModel().eval()
    rng = range(2, 20)
    M, K, N = choice(rng), choice(rng), choice(rng)  # noqa: N806
    fuzzed_x = (
        torch.rand(M, K, dtype=dtype)
        if dtype in {torch.bfloat16, torch.float16, torch.float32}
        else torch.randint(100, (M, K), dtype=dtype)
    )
    fuzzed_y = (
        torch.rand(K, N, dtype=dtype)
        if dtype in {torch.bfloat16, torch.float16, torch.float32}
        else torch.randint(100, (K, N), dtype=dtype)
    )

    if dynamic:
        k_dim = torch.export.Dim(name="K", min=1, max=32)
        dynamic_shapes = {"x": {1: k_dim}, "y": {0: k_dim}}
    else:
        dynamic_shapes = None

    await validate_numerical_output(
        model=model,
        custom_kernels=[custom_matmul],
        metal_inputs=True,
        input_names=["x", "y"],
        output_names=["result"],
        atol=0.1,
        rtol=0.1,
        dynamic_shapes=dynamic_shapes,
        x=fuzzed_x,
        y=fuzzed_y,
    )
