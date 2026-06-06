# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for softmax implementation."""

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

MAX_RANK = 8
MAX_DIM_SIZE = 20
# We can't exceed 1024 threads per thread group. We should probably have
# this be deferred to the runtime, but we're putting this guard in here
# so that the tests don't choose threadgroups that are too large.
MAX_THREADS = 1024


def torch_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Perform softmax across specified dimension."""
    return x.softmax(dim=dim)


@pytest.mark.skip(
    "reenable once runtime kernel moved to support Metal 4",
)
@pytest.mark.parametrize("positive", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
async def test_naive_softmax_kernel(
    positive: bool,
    dtype: torch.dtype,
    softmax_src: str,
) -> None:
    """We should be able to run a naive implementation of softmax."""
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

    class SoftmaxModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, axis: int) -> torch.Tensor:
            normalized_axis = x.dim() + axis if axis < 0 else axis
            axis_size = x.shape[normalized_axis]
            num_slices = x.numel() // axis_size
            return custom_softmax(
                x,
                axis,
                threads_per_grid=(num_slices, 1, 1),
                threads_per_thread_group=(min(MAX_THREADS, num_slices), 1, 1),
                result_shapes=[list(x.shape)],
            )

    model = SoftmaxModel().eval()
    rank = choice(range(2, 8))
    shape = [choice(range(1, MAX_DIM_SIZE)) for _ in range(rank)]
    fuzzed_input = torch.rand(*shape, dtype=dtype)
    scale = 1 if positive else -1
    dim = choice(range(rank)) * scale

    exported_model = torch.export.export(
        model,
        args=(fuzzed_input, dim),
    )
    ep = exported_model.run_decompositions(get_decomp_table())

    converter = TorchConverter()
    converter.register_custom_kernels([custom_softmax])
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
                    data=fuzzed_input.numpy(),
                    backing=StorageKind.METAL,
                ),
                "y": NDArray(
                    data=torch.tensor(dim, dtype=torch.int32).numpy(),
                    backing=StorageKind.METAL,
                ),
            },
        )
        result_arr = result["result"].numpy()
        np.testing.assert_array_almost_equal(
            result_arr,
            fuzzed_input.softmax(dim=dim).numpy(),
            decimal=2,
        )
