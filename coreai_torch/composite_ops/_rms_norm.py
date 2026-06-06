# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Torch implementation of composite root mean square normalization op."""

import torch
from typing_extensions import Self

from ._utils import Version


class RMSNormImpl(torch.nn.Module):
    """
    Core RMS normalization logic, intended to be externalized as a composite op.

    Takes both input and scale as explicit forward arguments so that both
    appear as graph inputs when externalized (rather than the scale being
    folded in as a constant from a sibling parameter).
    """

    def __init__(
        self: Self,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.axes = -1
        self.version = Version.v1

    def forward(self: Self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply root mean square normalization to input tensor."""
        # need f32, otherwise square may overflow f16 max 65504
        input_f32 = input.to(torch.float32)
        square_f32 = input_f32 * input_f32
        # need f32, otherwise accumulation may ignore small values
        mean_square_f32 = square_f32.mean(self.axes, keepdim=True)
        inv_rms_f32 = torch.rsqrt(mean_square_f32 + self.eps)
        input_normalized = input * inv_rms_f32

        # for the gemma3 case, the scale is always fp32, and hence we
        # do the down cast in the end
        if scale.dtype != input.dtype and scale.dtype == torch.float32:
            return (input_normalized * scale).to(input.dtype)

        # in other case, we convert the fp32 intermediate tensor back to
        # input dtype before multiplying with the scale
        return input_normalized.to(input.dtype) * scale


class RMSNorm(torch.nn.Module):
    """
    Apply root mean square normalization (RMSNorm) to input tensor, with attributes pre-determined.

    The RMSNorm operation is defined as:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale

    Where:
    - x is the input tensor
    - mean(x^2) is computed along the last dimension
    - eps is a small constant for numerical stability
    - scale is a learnable parameter, initialized to 1
    """

    def __init__(
        self: Self,
        dim: int,
        eps: float = 1e-5,
        n_heads: int | None = None,
    ) -> None:
        super().__init__()
        weight_shape: tuple[int, ...]
        if n_heads is None:
            # standard case: (dim,)
            weight_shape = (dim,)
        else:
            # multi-head case: (n_heads, 1, dim) for broadcasting
            # can be used to fuse query & key RMSNorms, where
            # - query_key has shape (batch_size, num_query_heads + num_key_heads, query_length, head_dim)
            # - n_heads = num_query_heads + num_key_heads
            # - dim = head_dim
            weight_shape = (n_heads, 1, dim)
        # as standard practice, initialize weight to 1
        weight = torch.ones(weight_shape)
        self.weight = torch.nn.Parameter(weight)
        self.rmsnorm_impl = RMSNormImpl(eps=eps)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Apply root mean square normalization to input tensor."""
        return self.rmsnorm_impl(x, self.weight)
