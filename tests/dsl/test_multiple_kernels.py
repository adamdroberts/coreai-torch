# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test for invoking multiple kernels in a single model."""

from pathlib import Path
from secrets import choice

import numpy as np
import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import NDArray, StorageKind

from coreai_torch import (
    MetalParameter,
    TorchConverter,
    TorchMetalKernel,
    get_decomp_table,
)

from ..utils import TemporaryModelAsset

pytestmark = pytest.mark.skip(
    reason="ExecutableOptions(enable_encoding_functions=...) was removed in the "
    "AIProgram API; no replacement found in coreai.authoring/runtime/compiler. "
    "DSL kernel tests need a follow-up once a replacement surfaces."
)


def torch_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Perform matrix multiplication on two input tensors."""
    return torch.matmul(x, y)


def torch_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Perform softmax across specified dimension."""
    return x.softmax(dim=dim)


@pytest.mark.skip(
    "reenable once runtime kernel moved to support Metal 4",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
async def test_multi_kernel_model(
    dtype: torch.dtype,
    float_tiled_matmul: str,
    softmax_src: str,
) -> None:
    """We should be able to write a model that chains multiple kernels."""
    custom_matmul = TorchMetalKernel(
        "tiled_matmul",
        input_names=["A", "B"],
        result_names=["C"],
        src=float_tiled_matmul,
        torch_defn=torch_matmul,
        metal_params=[
            MetalParameter("threadPos", "uint2", "thread_position_in_threadgroup"),
            MetalParameter("threadgroupPos", "uint2", "threadgroup_position_in_grid"),
            MetalParameter("threadsPerThreadgroup", "uint2", "threads_per_threadgroup"),
        ],
        template_dtypes={"A": "TYPE"},
    )

    custom_softmax = TorchMetalKernel(
        "custom_softmax",
        input_names=["input", "axis"],
        result_names=["output"],
        src=softmax_src,
        torch_defn=torch_softmax,
        metal_params=[
            MetalParameter("gid", "uint", "thread_position_in_grid"),
        ],
        template_dtypes={"input": "TYPE"},
    )

    class MultiKernelModel(torch.nn.Module):
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
                (N + TILE - 1) // TILE,
                (M + TILE - 1) // TILE,
                1,
            )
            threads = (TILE, TILE, 1)
            result_shape = [M, N]
            multiplied = custom_matmul(
                x,
                y,
                threads_per_grid=grid,
                threads_per_thread_group=threads,
                result_shapes=[result_shape],
            )
            num_slices = multiplied.numel() // multiplied.shape[1]
            return custom_softmax(
                multiplied,
                1,
                threads_per_grid=(num_slices, 1, 1),
                threads_per_thread_group=(num_slices, 1, 1),
                result_shapes=[list(multiplied.shape)],
            )

    model = MultiKernelModel().eval()
    rng = range(2, 40)
    M, K, N = choice(rng), choice(rng), choice(rng)  # noqa: N806
    fuzzed_x = torch.rand(M, K, dtype=dtype)
    fuzzed_y = torch.rand(K, N, dtype=dtype)
    exported_model = torch.export.export(
        model,
        args=(fuzzed_x, fuzzed_y),
    )
    ep = exported_model.run_decompositions(get_decomp_table())

    converter = TorchConverter()
    converter.register_custom_kernels([custom_matmul, custom_softmax])
    converter.add_exported_program(
        ep,
        input_names=["x", "y"],
        output_names=["result"],
    )
    coreai_program = converter.to_coreai()

    compile_options = AIProgram.ExecutableOptions(
        enable_encoding_functions=True,
    )
    with TemporaryModelAsset() as tmp:
        ai_model = await coreai_program.create_aimodel(
            Path(tmp), options=compile_options
        )
        function = ai_model.load_function("main")
        result = await function(
            {
                "x": NDArray(
                    data=fuzzed_x.numpy(),
                    backing=StorageKind.METAL,
                ),
                "y": NDArray(
                    data=fuzzed_y.numpy(),
                    backing=StorageKind.METAL,
                ),
            },
        )
        result_arr = result["result"].numpy()
        np.testing.assert_array_almost_equal(
            result_arr,
            torch.matmul(fuzzed_x, fuzzed_y).softmax(dim=1).numpy(),
            decimal=2,
        )
