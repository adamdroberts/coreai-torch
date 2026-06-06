# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""SubModel for hierarchical benchmarking tests."""

import torch


class SubModel(torch.nn.Module):
    """A sub-module that performs some operations."""

    def __init__(self) -> None:
        """Initialize the sub-model."""
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sub-model."""
        x = self.linear(x)
        x = torch.relu(x)
        return x
