# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for metal failures."""

import re
import sys
from pathlib import Path

import pytest
import torch
from coreai.runtime import AIModel

from coreai_torch import TorchConverter, TorchMetalKernel, get_decomp_table

from ..utils import TemporaryModelAsset


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests run only on Mac")
async def test_raise_comprehensible_compilation_failure() -> None:
    """We should raise readable compilation failures."""

    def torch_fn(x: torch.Tensor) -> torch.Tensor:
        return x

    custom_kernel = TorchMetalKernel(
        "bad_operation",
        input_names=["x"],
        result_names=["output"],
        src="A[s] = sdfs",
        torch_defn=torch_fn,
    )

    class MetalModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return custom_kernel(
                x,
                threads_per_grid=(x.shape[0], 1, 1),
                threads_per_thread_group=(1, 1, 1),
                result_shapes=[list(x.shape)],
            )

    model = MetalModel().eval()
    exported_model = torch.export.export(
        model,
        args=(torch.rand(10, dtype=torch.float16),),
    )
    ep = exported_model.run_decompositions(get_decomp_table())

    converter = TorchConverter()
    converter.register_custom_kernels([custom_kernel])
    converter.add_exported_program(
        ep,
        input_names=["x"],
        output_names=["result"],
    )
    coreai_program = converter.to_coreai()

    with TemporaryModelAsset() as tmp:
        coreai_program.save_asset(Path(tmp))
        model = await AIModel.load(Path(tmp))
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Kernel coreai.metal4_kernel invoked with invalid parameters",
            ),
        ):
            model.load_function("main")
