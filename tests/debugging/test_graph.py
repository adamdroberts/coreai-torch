# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for computation graph creation and depth calculation."""

import torch

from coreai_torch.debugging.graph import create_graph_from_exported_program

from .test_model import LinearReluMulModel, TwoLinearSigmoidModel, get_example_inputs


def test_create_graph_basic() -> None:
    """Test basic graph creation from exported program."""
    model = LinearReluMulModel().eval()
    example_inputs = get_example_inputs(LinearReluMulModel)
    args = tuple(example_inputs.values())

    exported_program = torch.export.export(model, args=args)
    graph = create_graph_from_exported_program(exported_program)

    # Verify graph structure
    assert graph is not None
    assert graph.original_graph == exported_program.graph

    # Verify nodes exist
    nodes = graph.get_nodes()
    assert len(nodes) > 0

    # Verify all nodes have valid properties
    for node in nodes:
        assert node.op_id is not None
        assert node.original_node is not None
        assert isinstance(node.predecessors, list)
        assert isinstance(node.depth, int)
        assert node.depth >= 0


def test_graph_node_retrieval() -> None:
    """Test node retrieval by ID."""
    model = LinearReluMulModel().eval()
    example_inputs = get_example_inputs(LinearReluMulModel)
    args = tuple(example_inputs.values())

    exported_program = torch.export.export(model, args=args)
    graph = create_graph_from_exported_program(exported_program)

    # Get all operation IDs
    op_ids = graph.get_op_ids()
    assert len(op_ids) > 0

    # Verify we can retrieve each node by its ID
    for op_id in op_ids:
        node = graph.get_node_by_id(op_id)
        assert node is not None
        assert node.op_id == op_id


def test_depth_calculation() -> None:
    """Test that depth is calculated correctly based on dependencies."""
    model = TwoLinearSigmoidModel().eval()
    example_inputs = get_example_inputs(TwoLinearSigmoidModel)
    args = tuple(example_inputs.values())

    exported_program = torch.export.export(model, args=args)
    graph = create_graph_from_exported_program(exported_program)

    nodes = graph.get_nodes()

    # Verify depth increases as we progress through the graph
    # Input nodes should have lower depth than output nodes
    depths = [node.depth for node in nodes]

    # Check that we have a range of depths (not all the same)
    assert min(depths) >= 0
    assert max(depths) > min(depths), "Depths should vary across the graph"

    # Verify predecessor relationship: predecessors should have lower or equal depth
    for node in nodes:
        for pred_id in node.predecessors:
            pred_node = graph.get_node_by_id(pred_id)
            assert pred_node.depth <= node.depth, (
                f"Predecessor {pred_id} (depth {pred_node.depth}) should have "
                f"depth <= successor {node.op_id} (depth {node.depth})"
            )


def test_nodes_by_depth() -> None:
    """Test grouping nodes by depth level."""
    model = TwoLinearSigmoidModel().eval()
    example_inputs = get_example_inputs(TwoLinearSigmoidModel)
    args = tuple(example_inputs.values())

    exported_program = torch.export.export(model, args=args)
    graph = create_graph_from_exported_program(exported_program)

    # Get nodes grouped by depth
    depth_map = graph.get_nodes_by_depth()

    # Verify structure
    assert isinstance(depth_map, dict)
    assert len(depth_map) > 0

    # Verify all depths are non-negative
    assert all(depth >= 0 for depth in depth_map.keys())

    # Verify all nodes at each depth level
    for depth, nodes_at_depth in depth_map.items():
        assert len(nodes_at_depth) > 0
        for node in nodes_at_depth:
            assert node.depth == depth


def test_predecessors_extracted() -> None:
    """Test that predecessors are correctly extracted."""
    model = LinearReluMulModel().eval()
    example_inputs = get_example_inputs(LinearReluMulModel)
    args = tuple(example_inputs.values())

    exported_program = torch.export.export(model, args=args)
    graph = create_graph_from_exported_program(exported_program)

    nodes = graph.get_nodes()

    # Find nodes with predecessors
    nodes_with_preds = [n for n in nodes if len(n.predecessors) > 0]

    # Should have some nodes with predecessors
    assert len(nodes_with_preds) > 0

    # Verify predecessor IDs are valid
    all_op_ids = set(graph.get_op_ids())
    for node in nodes_with_preds:
        for pred_id in node.predecessors:
            assert pred_id in all_op_ids, (
                f"Predecessor {pred_id} not found in graph's operation IDs"
            )


def test_original_node_reference() -> None:
    """Test that original FX nodes are preserved."""
    model = LinearReluMulModel().eval()
    example_inputs = get_example_inputs(LinearReluMulModel)
    args = tuple(example_inputs.values())

    exported_program = torch.export.export(model, args=args)
    graph = create_graph_from_exported_program(exported_program)

    # Verify original nodes are preserved
    nodes = graph.get_nodes()
    fx_nodes = list(exported_program.graph.nodes)

    # Create mapping by name
    fx_node_map = {fx_node.name: fx_node for fx_node in fx_nodes}

    # Verify each graph node has the correct original FX node
    for node in nodes:
        assert node.op_id in fx_node_map
        assert node.original_node == fx_node_map[node.op_id]


def test_subgraph_extraction() -> None:
    """Test extracting a subgraph."""
    model = TwoLinearSigmoidModel().eval()
    example_inputs = get_example_inputs(TwoLinearSigmoidModel)
    args = tuple(example_inputs.values())

    exported_program = torch.export.export(model, args=args)
    graph = create_graph_from_exported_program(exported_program)

    nodes = graph.get_nodes()
    total_nodes = len(nodes)

    # Extract middle portion
    start, end = total_nodes // 4, 3 * total_nodes // 4
    subgraph = graph.get_subgraph(start, end)

    # Verify subgraph
    assert len(subgraph) == end - start
    for i, node in enumerate(subgraph):
        assert node == nodes[start + i]
