# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for causal self-attention kernel."""

import math
from pathlib import Path

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

# Metal source for causal (masked) self-attention.
# compute the forward pass of the masked self-attention without an
# explicit mask array.
# Inputs  : X (Txd), Wq (dxd), Wk (dxd), Wv (dxd) — row-major float32.
# Output  : output (Txd) — row-major float32.
# Dispatch: 1-D grid of T threads; thread `gid` computes output row `gid`.
#
# Per-thread algorithm:
#   1. Q[gid, :]      = X[gid, :] @ Wq
#   2. score[col]     = dot(Q[gid], X[col, :] @ Wk) / sqrt(d)   col <= gid
#                     = -inf                                       col >  gid  (causal mask)
#   3. attn[col]      = softmax(scores)[col]
#   4. output[gid, :] = sum_col  attn[col] * (X[col, :] @ Wv)
#
# MAX_T / MAX_D bound the grid; keep T and d within these limits in tests.

SELF_ATTENTION_SRC = """
constexpr uint MAX_T = 64;
constexpr uint MAX_D = 64;

const uint row = gid;
const uint T   = X.get_extent(0);
const uint d   = X.get_extent(1);

if (row >= T) return;

const float inv_sqrt_d = 1.0f / sqrt(float(d));

// --- Step 1: Q[row, :] = X[row, :] @ Wq ---
float Q[MAX_D];
for (uint j = 0; j < d; ++j) {
    float acc = 0.0f;
    for (uint k = 0; k < d; ++k) {
        acc += X[row, k] * Wq[k, j];
    }
    Q[j] = acc;
}

// --- Step 2: scores[col] = dot(Q[row], K[col]) * inv_sqrt_d, causal mask ---
// K[col, :] = X[col, :] @ Wk is computed on the fly to avoid a MAX_T x MAX_D buffer.
float scores[MAX_T];
for (uint col = 0; col < T; ++col) {
    if (col > row) {
        scores[col] = -INFINITY;   // upper-triangular causal mask
        continue;
    }
    float acc = 0.0f;
    for (uint j = 0; j < d; ++j) {
        float K_col_j = 0.0f;
        for (uint k = 0; k < d; ++k) {
            K_col_j += X[col, k] * Wk[k, j];
        }
        acc += Q[j] * K_col_j;
    }
    scores[col] = acc * inv_sqrt_d;
}

// --- Step 3: numerically-stable softmax over scores ---
// max only over unmasked positions (col <= row); masked positions are -inf.
float max_val = scores[0];
for (uint col = 1; col <= row; ++col) {
    max_val = max(max_val, scores[col]);
}
float sum_exp = 0.0f;
for (uint col = 0; col < T; ++col) {
    // exp(-inf - max_val) == 0, so masked positions contribute nothing.
    sum_exp += exp(scores[col] - max_val);
}

// --- Step 4: output[row, :] = attn @ V,  V[col, :] = X[col, :] @ Wv ---
for (uint j = 0; j < d; ++j) {
    float acc = 0.0f;
    for (uint col = 0; col <= row; ++col) {   // col > row has attn == 0
        const float attn_col = exp(scores[col] - max_val) / sum_exp;
        float V_col_j = 0.0f;
        for (uint k = 0; k < d; ++k) {
            V_col_j += X[col, k] * Wv[k, j];
        }
        acc += attn_col * V_col_j;
    }
    output[row, j] = acc;
}
"""


def torch_self_attention(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
) -> torch.Tensor:
    """Causal self-attention reference implemented in PyTorch."""
    t, d = x.shape
    q = x @ wq  # (T, d)
    k = x @ wk  # (T, d)
    v = x @ wv  # (T, d)
    scores = (q @ k.T) / math.sqrt(d)  # (T, T)
    # Upper-triangular causal mask: positions col > row are set to -inf.
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)  # (T, T)
    return attn @ v  # (T, d)


@pytest.mark.skip(
    "reenable once runtime kernel moved to support Metal 4",
)
async def test_causal_self_attention_kernel() -> None:
    """Causal self-attention should match the PyTorch reference for T=8, d=8."""
    custom_attention = TorchMetalKernel(
        "self_attention",
        input_names=["X", "Wq", "Wk", "Wv"],
        result_names=["output"],
        src=SELF_ATTENTION_SRC,
        torch_defn=torch_self_attention,
        metal_params=[MetalParameter("gid", "uint", "thread_position_in_grid")],
    )

    t, d = 8, 8

    class SelfAttentionModel(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            wq: torch.Tensor,
            wk: torch.Tensor,
            wv: torch.Tensor,
        ) -> torch.Tensor:
            return custom_attention(
                x,
                wq,
                wk,
                wv,
                threads_per_grid=(x.shape[0], 1, 1),
                threads_per_thread_group=(1, 1, 1),
                result_shapes=[[x.shape[0], x.shape[1]]],
            )

    model = SelfAttentionModel().eval()
    x = torch.rand(t, d, dtype=torch.float32)
    wq = torch.rand(d, d, dtype=torch.float32)
    wk = torch.rand(d, d, dtype=torch.float32)
    wv = torch.rand(d, d, dtype=torch.float32)

    exported_model = torch.export.export(model, args=(x, wq, wk, wv))
    ep = exported_model.run_decompositions(get_decomp_table())

    converter = TorchConverter()
    converter.register_custom_kernels([custom_attention])
    converter.add_exported_program(
        ep,
        input_names=["x", "wq", "wk", "wv"],
        output_names=["output"],
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
                "x": NDArray(data=x.numpy(), backing=StorageKind.METAL),
                "wq": NDArray(data=wq.numpy(), backing=StorageKind.METAL),
                "wk": NDArray(data=wk.numpy(), backing=StorageKind.METAL),
                "wv": NDArray(data=wv.numpy(), backing=StorageKind.METAL),
            },
        )
        result_arr = result["output"].numpy()
        expected = torch_self_attention(x, wq, wk, wv).numpy()
        np.testing.assert_array_almost_equal(result_arr, expected, decimal=4)
