# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Torch implementation of composite gather matmul op."""

import torch
from torch import Tensor

from ._utils import Version


def _gather(x: Tensor, indices: Tensor, num_batch_axes: int = 0) -> Tensor:
    x_shape = x.shape
    result_shape = (
        x_shape[0:num_batch_axes] + indices.shape + x_shape[num_batch_axes + 1 :]
    )
    # TODO: Remove this explict cast once torch supports uint indices
    indices = indices.to(torch.int32)  # type: ignore[assignment]

    flat_indices = indices.flatten()
    flat_gather = torch.index_select(x, dim=num_batch_axes, index=flat_indices)
    result = flat_gather.view(result_shape)
    return result


class GatherMM(torch.nn.Module):
    """
    Gather matmul composite op.

    Optionally gathers rows from lhs and/or rhs using the provided index
    tensors, then performs a matmul on the gathered operands.

    (Rough) formulation:
        if lhs_indices: lhs = gather(lhs, lhs_indices)
        if rhs_indices: rhs = gather(rhs, rhs_indices)
        return matmul(lhs, rhs)
    """

    def __init__(self, num_batch_axes: int = 0) -> None:
        super().__init__()
        self.num_batch_axes = int(num_batch_axes)
        self.version = Version.v1

    def forward(
        self,
        lhs: Tensor,
        rhs: Tensor,
        lhs_indices: Tensor | None = None,
        rhs_indices: Tensor | None = None,
    ) -> Tensor:
        """Perform gather matmul."""
        if lhs_indices is not None:
            lhs = _gather(lhs, lhs_indices, num_batch_axes=self.num_batch_axes)
        if rhs_indices is not None:
            rhs = _gather(rhs, rhs_indices, num_batch_axes=self.num_batch_axes)
        return torch.matmul(lhs, rhs)
