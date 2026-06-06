# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Test for lowering a simple PyTorch model to Core AI format.
"""

import torch
import torch.nn as nn

from coreai_torch import TorchConverter


class SimpleModel(nn.Module):
    """A simple PyTorch model for testing lowering."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def test_lower_simple_model():
    """Test lowering a simple PyTorch model to Core AI format."""
    # Create and prepare the model
    model = SimpleModel()
    model.eval()

    # Create example input
    example_input = (torch.randn(1, 10),)

    # Export the PyTorch model
    exported_program = torch.export.export(model, args=example_input)

    # Verify the exported program is valid
    assert exported_program is not None
    assert hasattr(exported_program, "graph")

    # Run decompositions before converting
    exported_program = exported_program.run_decompositions()

    # Convert to Core AI using TorchConverter
    converter = TorchConverter().add_exported_program(
        exported_program,
        input_names=("x",),
        output_names=("out",),
    )
    coreai_program = converter.to_coreai()

    # Verify the Core AI program was created
    assert coreai_program is not None
