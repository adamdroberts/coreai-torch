# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test for _get_module_hierarchy function in converter.py"""

import torch

from coreai_torch._utils import _get_module_hierarchy, _ModuleInstanceRegistry


class Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        return self.act(self.linear(x))


class SmallModel(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.block = Block(dim)  # one submodule
        self.out = torch.nn.Linear(dim, 2)

    def forward(self, x):
        x = self.block(x)  # first call
        x = self.block(x)  # second back-to-back call
        x = self.out(x)
        return x


def test_get_module_hierarchy_distinguishes_back_to_back_calls():
    """
    Test that _get_module_hierarchy assigns different occurrence indices to repeated calls.
    """
    model = SmallModel(dim=8).eval()
    x = torch.randn(2, 8)
    ep = torch.export.export(model, args=(x,))
    registry = _ModuleInstanceRegistry()

    block_occurrences = set()
    linear_occurrences = set()
    relu_occurrences = set()

    for node in ep.graph.nodes:
        if node.op == "call_function" and "nn_module_stack" in node.meta:
            hierarchy = _get_module_hierarchy(node=node, registry=registry)
            # Find hierarchies containing Block
            for name in hierarchy:
                if "Block" in name:
                    block_occurrences.add(name)
                elif "Linear" in name:
                    linear_occurrences.add(name)
                elif "ReLU" in name:
                    relu_occurrences.add(name)

    # We expect two different occurrence indices for the two Block calls
    assert len(block_occurrences) >= 2, (
        f"Expected at least 2 different occurrence indices for Block calls, got {block_occurrences}"
    )

    # We expect three different occurrence indices for the Linear calls
    assert len(linear_occurrences) >= 3, (
        f"Expected at least 3 different occurrence indices for Linear calls, got {linear_occurrences}"
    )

    # We expect two different occurrence indices for the ReLU calls
    assert len(relu_occurrences) >= 2, (
        f"Expected at least 2 different occurrence indices for ReLU calls, got {relu_occurrences}"
    )
