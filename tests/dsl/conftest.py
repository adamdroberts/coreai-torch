# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Fixtures for Metal kernel integration tests."""

import sys
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark DSL tests as 'dsl', add flaky reruns, and skip on non-macOS."""
    for item in items:
        if Path(item.fspath).is_relative_to(_THIS_DIR):
            item.add_marker(pytest.mark.dsl)
            item.add_marker(pytest.mark.flaky(reruns=3))
            if sys.platform != "darwin":
                item.add_marker(pytest.mark.skip(reason="Metal tests run only on Mac"))


# ---------------------------------------------------------------------------
# Metal kernel source fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def softmax_src() -> str:
    """Fixture for softmax kernel."""
    return """
        // We're allowing for users to specify axes as negative, so we need to
        // normalize the axis to its absolute value.
        uint normalized_axis = axis >= 0 ? uint(axis) : uint(input.get_rank() + axis);

        // Maximum supported rank (Metal doesn't allow VLAs, so we use a
        // fixed-size array; 8 covers all practical use-cases).
        constexpr uint MAX_RANK = 8;

        // Return if we exceed the maximum rank.
        if (input.get_rank() > MAX_RANK) return;

        // Total number of slices = total elements / axis_size.
        // Each thread is responsible for exactly one slice.
        uint axis_size = input.get_extent(normalized_axis);

        // Compute total number of slices to guard against over-dispatch.
        uint num_slices = 1;
        for (uint d = 0; d < input.get_rank(); ++d) {
            if (d != normalized_axis) num_slices *= input.get_extent(d);
        }
        if (gid >= num_slices) return;

        // ------------------------------------------------------------------
        // Recover the N-D coordinates for this slice.
        //
        // We treat the tensor as if the axis dimension were removed, giving a
        // "collapsed" shape of rank-1 dimensions. We convert `gid` into
        // coordinates in that collapsed space, then re-insert a placeholder
        // (0) for the axis dimension to get a base coordinate vector.
        // ------------------------------------------------------------------

        // Build the collapsed shape (rank-1 dims, axis removed).
        uint collapsed_shape[MAX_RANK];
        uint collapsed_rank = 0;
        for (uint d = 0; d < input.get_rank(); ++d) {
            if (d != normalized_axis) collapsed_shape[collapsed_rank++] = input.get_extent(d);
        }

        // Decode gid into collapsed coordinates.
        uint collapsed_coords[MAX_RANK];
        {
            uint remaining = gid;
            for (int d = int(collapsed_rank) - 1; d >= 0; --d) {
                collapsed_coords[d] = remaining % collapsed_shape[d];
                remaining           = remaining / collapsed_shape[d];
            }
        }

        // Re-insert the axis slot (value 0 as a base; we'll vary it in the loop).
        uint coords[MAX_RANK];
        {
            uint ci = 0; // index into collapsed_coords
            for (uint d = 0; d < input.get_rank(); ++d) {
                coords[d] = (d == normalized_axis) ? 0 : collapsed_coords[ci++];
            }
        }

        // ------------------------------------------------------------------
        // Pass 1 - find maximum value along the axis (numerical stability)
        // ------------------------------------------------------------------
        TYPE max_val = -INFINITY;
        for (uint i = 0; i < axis_size; ++i) {
            coords[normalized_axis] = i;
            metal::array<uint, input.get_rank()> idx;
            for (uint d = 0; d < input.get_rank(); ++d) idx[d] = coords[d];
            max_val = max(max_val, input[idx]);
        }

        // ------------------------------------------------------------------
        // Pass 2 - sum of exp(x - max)
        // ------------------------------------------------------------------
        TYPE sum_exp = 0.0f;
        for (uint i = 0; i < axis_size; ++i) {
            coords[normalized_axis] = i;
            metal::array<uint, input.get_rank()> idx;
            for (uint d = 0; d < input.get_rank(); ++d) idx[d] = coords[d];
            sum_exp += exp(input[idx] - max_val);
        }

        // ------------------------------------------------------------------
        // Pass 3 - write normalised output
        // ------------------------------------------------------------------
        for (uint i = 0; i < axis_size; ++i) {
            coords[normalized_axis] = i;
            metal::array<uint, input.get_rank()> idx;
            for (uint d = 0; d < input.get_rank(); ++d) idx[d] = coords[d];
            output[idx] = exp(input[idx] - max_val) / sum_exp;
        }
    """


@pytest.fixture(scope="module")
def generic_naive_matmul() -> str:
    """Fixture for unspecialized matmul."""
    return """
        const uint K = A.get_extent(0);
        const uint M = A.get_extent(1);
        const uint N = B.get_extent(0);

        if (gid.x >= N || gid.y >= M) return; // bounds guard

        TYPE sum = ZERO;
        for (uint k = 0; k < K; ++k) {
            sum += A[k, gid.y] * B[gid.x, k];
        }
        C[gid.x, gid.y] = sum;
    """


@pytest.fixture(scope="module")
def int_naive_matmul(generic_naive_matmul: str) -> str:
    """Fixture for matmul specialized for integer inputs."""
    return generic_naive_matmul.replace("ZERO", "0")


@pytest.fixture(scope="module")
def float_naive_matmul(generic_naive_matmul: str) -> str:
    """Fixture for matmul specialized for float inputs."""
    return generic_naive_matmul.replace("ZERO", "0.0f")


@pytest.fixture(scope="module")
def bfloat_naive_matmul(generic_naive_matmul: str) -> str:
    """Fixture for matmul specialized for float inputs."""
    return generic_naive_matmul.replace("ZERO", "bfloat(0.0)")


@pytest.fixture(scope="module")
def generic_tiled_matmul() -> str:
    """Fixture for unspecialized tiled matmul."""
    return """
        const uint TILE = 16;
        // A shape [K,M]: A[k, m]   B shape [N,K]: B[n, k]   C shape [N,M]: C[n, m]
        const uint K = A.get_extent(0);
        const uint M = A.get_extent(1);
        const uint N = B.get_extent(0);

        threadgroup TYPE tileA[TILE][TILE];  // tileA[row_in_tile][k_in_tile]
        threadgroup TYPE tileB[TILE][TILE];  // tileB[k_in_tile][col_in_tile]

        TYPE accum = ZERO;
        const uint numTiles = (K + TILE - 1) / TILE;

        for (uint t = 0; t < numTiles; ++t) {
            // Each thread loads one element of the A-tile and one of the B-tile.
            const uint a_k = t * TILE + tid.x;
            const uint a_m = tgid.y * TILE + tid.y;
            tileA[tid.y][tid.x] = (a_k < K && a_m < M) ? A[a_k, a_m] : ZERO;

            const uint b_n = tgid.x * TILE + tid.x;
            const uint b_k = t * TILE + tid.y;
            tileB[tid.y][tid.x] = (b_n < N && b_k < K) ? B[b_n, b_k] : ZERO;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0; kk < TILE; ++kk) {
                accum += tileA[tid.y][kk] * tileB[kk][tid.x];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (gid.x < N && gid.y < M) {
            C[gid.x, gid.y] = accum;
        }
    """


@pytest.fixture(scope="module")
def int_tiled_matmul(generic_tiled_matmul: str) -> str:
    """Fixture for tiled matmul specialized for integer inputs."""
    return generic_tiled_matmul.replace("ZERO", "0")


@pytest.fixture(scope="module")
def float_tiled_matmul(generic_tiled_matmul: str) -> str:
    """Fixture for tiled matmul specialized for float inputs."""
    return generic_tiled_matmul.replace("ZERO", "0.0f")


@pytest.fixture(scope="module")
def bfloat_tiled_matmul(generic_tiled_matmul: str) -> str:
    """Fixture for tiled matmul specialized for bfloat16 inputs."""
    return generic_tiled_matmul.replace("ZERO", "bfloat(0.0)")
