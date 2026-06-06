# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test custom layers."""

import numpy as np
import pytest
import torch
import transformers

from coreai_torch._compression._floatx import Float4Tensor
from coreai_torch._compression._intx import SubbyteTensor
from coreai_torch._compression.custom_layers import (
    constexpr_blockwise_shift_scale,
    dequantize,
    quantize,
    sparse_to_dense,
)
from coreai_torch._compression.utils import _inject_subbyte_in_quant

# ──────────────────────────────────────────────────────────────────────
# constexpr_blockwise_shift_scale
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="fp4 weight dequant via custom op not yet supported")
def test_blockwise_shift_scale_fp4_matches_huggingface() -> None:
    """Test FP4/MX dequantization matches HuggingFace convert_moe_packed_tensors."""
    num_experts = 32
    output_dims = 576
    input_dims = 288
    block_size = 32
    blocks = torch.randint(
        0,
        255,
        (num_experts, output_dims, input_dims // block_size, block_size // 2),
        dtype=torch.uint8,
    )
    scales = torch.randint(
        0,
        255,
        (num_experts, output_dims, input_dims // block_size),
        dtype=torch.uint8,
    )

    blocks_coreai = Float4Tensor(blocks.flatten(-2))
    scales_coreai = scales.view(torch.float8_e8m0fnu)
    dequantized_coreai = constexpr_blockwise_shift_scale(
        blocks_coreai,
        scales_coreai,
        output_dtype=torch.float32,
    )

    dequantized_hf = (
        transformers.integrations.mxfp4.convert_moe_packed_tensors(
            blocks,
            scales,
            dtype=torch.float32,
        )
        .transpose(-1, -2)
        .contiguous()
    )
    np.testing.assert_allclose(dequantized_coreai, dequantized_hf)


@pytest.mark.parametrize(
    (
        "input_t",
        "scale",
        "zero_point",
        "minval",
        "input_dtype",
        "output_dtype",
        "match",
    ),
    [
        pytest.param(
            torch.randint(0, 10, (4, 8), dtype=torch.int8),
            torch.rand(4, dtype=torch.float32),
            None,
            None,
            None,
            None,
            "rank.*mismatch",
            id="rank_mismatch",
        ),
        pytest.param(
            torch.randint(0, 10, (4, 8), dtype=torch.int8),
            torch.rand(2, 3, dtype=torch.float32),
            None,
            None,
            None,
            None,
            "not element-wise divisible",
            id="scale_not_divisible",
        ),
        pytest.param(
            torch.randint(0, 10, (4, 4), dtype=torch.int8),
            torch.rand(4, 4, dtype=torch.float32),
            torch.zeros(4, 4, dtype=torch.float32),
            None,
            None,
            None,
            "zero_point dtype.*must match input dtype",
            id="int_zero_point_dtype_mismatch",
        ),
        pytest.param(
            torch.randint(0, 10, (4, 4), dtype=torch.int8),
            torch.rand(2, 2, dtype=torch.float32),
            torch.zeros(4, 4, dtype=torch.int8),
            None,
            None,
            None,
            "zero_point shape.*scale shape.*mismatch",
            id="int_zero_point_shape_mismatch",
        ),
        pytest.param(
            torch.randint(0, 10, (4, 4), dtype=torch.int8),
            torch.rand(2, 2, dtype=torch.float32),
            None,
            torch.zeros(4, 4, dtype=torch.float32),
            torch.int8,
            None,
            "minval shape.*scale shape.*mismatch",
            id="int_minval_shape_mismatch",
        ),
        pytest.param(
            torch.randint(0, 10, (4, 4), dtype=torch.int8),
            torch.rand(2, 2, dtype=torch.float32),
            None,
            torch.zeros(2, 2, dtype=torch.float16),
            torch.int8,
            None,
            "minval dtype.*must match scale dtype",
            id="int_minval_dtype_mismatch",
        ),
        pytest.param(
            torch.randint(0, 10, (4, 4), dtype=torch.int8),
            torch.rand(2, 2, dtype=torch.float32),
            None,
            torch.zeros(2, 2, dtype=torch.float32),
            None,
            None,
            "input_dtype is required when minval is provided",
            id="int_minval_missing_input_dtype",
        ),
        pytest.param(
            torch.randint(0, 10, (4, 4), dtype=torch.int8),
            torch.rand(4, 4, dtype=torch.float32),
            torch.zeros(4, 4, dtype=torch.int8),
            torch.zeros(4, 4, dtype=torch.float32),
            torch.int8,
            None,
            "zero_point and minval are mutually exclusive",
            id="int_both_zp_and_minval",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0], dtype=torch.float8_e5m2),
            torch.ones(2, dtype=torch.float8_e8m0fnu),
            None,
            None,
            None,
            None,
            "output_dtype is required for FP dequantization with e8m0fnu",
            id="fp_e8m0_missing_output_dtype",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0], dtype=torch.float8_e5m2),
            torch.ones(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float8_e5m2),
            None,
            None,
            None,
            "zero_point is not supported for FP",
            id="fp_with_zero_point",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0], dtype=torch.float8_e5m2),
            torch.ones(2, dtype=torch.float32),
            None,
            torch.zeros(2, dtype=torch.float32),
            None,
            None,
            "minval is not supported for FP",
            id="fp_with_minval",
        ),
    ],
)
def test_blockwise_shift_scale_validation(  # noqa: PLR0913
    input_t: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    minval: torch.Tensor | None,
    input_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
    match: str,
) -> None:
    """Validate that invalid inputs raise ValueError."""
    with pytest.raises(ValueError, match=match):
        constexpr_blockwise_shift_scale(
            input_t,
            scale,
            zero_point=zero_point,
            minval=minval,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )


# ──────────────────────────────────────────────────────────────────────
# quantize — validation tests
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("input_t", "scale", "output_dtype", "zero_point", "minval", "match"),
    [
        pytest.param(
            torch.rand(4, 4, dtype=torch.float32),
            torch.rand(3, dtype=torch.float32),
            torch.int8,
            None,
            None,
            "cannot be reshaped to 0-D or 1-D",
            id="scale_numel_mismatch",
        ),
        pytest.param(
            torch.rand(4, 4, dtype=torch.float32),
            torch.rand(4, 1, 1, dtype=torch.float32),
            torch.int8,
            None,
            None,
            "scale rank.*must match input rank",
            id="scale_rank_mismatch",
        ),
        pytest.param(
            torch.rand(4, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
            torch.int8,
            torch.zeros(4, dtype=torch.int8),
            None,
            "zero_point shape.*scale shape.*mismatch",
            id="zero_point_shape_mismatch",
        ),
        pytest.param(
            torch.rand(4, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
            torch.int8,
            None,
            torch.zeros(4, dtype=torch.float32),
            "minval shape.*scale shape.*mismatch",
            id="minval_shape_mismatch",
        ),
        pytest.param(
            torch.rand(4, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
            torch.int8,
            None,
            torch.tensor(0.0, dtype=torch.float16),
            "minval dtype.*must match input dtype",
            id="minval_dtype_mismatch",
        ),
        pytest.param(
            torch.rand(4, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
            torch.int8,
            torch.tensor(0, dtype=torch.int8),
            torch.tensor(0.0, dtype=torch.float32),
            "zero_point and minval are mutually exclusive",
            id="both_zp_and_minval",
        ),
        pytest.param(
            torch.rand(4, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
            torch.float8_e5m2,
            torch.tensor(0.0, dtype=torch.float8_e5m2),
            None,
            "zero_point is not supported for FP",
            id="fp_with_zero_point",
        ),
        pytest.param(
            torch.rand(4, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
            torch.float8_e5m2,
            None,
            torch.tensor(0.0, dtype=torch.float32),
            "minval is not supported for FP",
            id="fp_with_minval",
        ),
    ],
)
def test_quantize_validation(  # noqa: PLR0913
    input_t: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype,
    zero_point: torch.Tensor | None,
    minval: torch.Tensor | None,
    match: str,
) -> None:
    """Validate that invalid inputs raise ValueError."""
    with pytest.raises(ValueError, match=match):
        quantize(
            input_t,
            scale,
            output_dtype,
            zero_point=zero_point,
            minval=minval,
        )


# ──────────────────────────────────────────────────────────────────────
# dequantize — validation tests
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    (
        "input_t",
        "scale",
        "zero_point",
        "minval",
        "input_dtype",
        "output_dtype",
        "match",
    ),
    [
        pytest.param(
            torch.randint(0, 10, (4, 4), dtype=torch.int8),
            torch.rand(3, dtype=torch.float32),
            None,
            None,
            None,
            None,
            "cannot be reshaped to 0-D or 1-D",
            id="scale_numel_mismatch",
        ),
        pytest.param(
            torch.randint(0, 10, (4, 4), dtype=torch.int8),
            torch.rand(4, 1, 1, dtype=torch.float32),
            None,
            None,
            None,
            None,
            "scale rank.*must match input rank",
            id="scale_rank_mismatch",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0], dtype=torch.float8_e5m2),
            torch.ones(2, dtype=torch.float8_e8m0fnu),
            None,
            None,
            None,
            None,
            "output_dtype is required for FP dequantization with e8m0fnu",
            id="fp_e8m0_missing_output_dtype",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0], dtype=torch.float8_e5m2),
            torch.tensor(1.0, dtype=torch.float32),
            torch.ones(1, dtype=torch.float8_e5m2),
            None,
            None,
            None,
            "zero_point is not supported for FP",
            id="fp_with_zero_point",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0], dtype=torch.float8_e5m2),
            torch.tensor(1.0, dtype=torch.float32),
            None,
            torch.zeros(1, dtype=torch.float32),
            None,
            None,
            "minval is not supported for FP",
            id="fp_with_minval",
        ),
        pytest.param(
            torch.randint(0, 10, (4,), dtype=torch.int8),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            None,
            None,
            None,
            "zero_point dtype.*must match input dtype",
            id="int_zero_point_dtype_mismatch",
        ),
        pytest.param(
            torch.randint(0, 10, (4,), dtype=torch.int8),
            torch.tensor(1.0, dtype=torch.float32),
            torch.zeros(4, dtype=torch.int8),
            None,
            None,
            None,
            "zero_point shape.*scale shape.*mismatch",
            id="int_zero_point_shape_mismatch",
        ),
        pytest.param(
            torch.randint(0, 10, (4,), dtype=torch.int8),
            torch.tensor(1.0, dtype=torch.float32),
            None,
            torch.zeros(4, dtype=torch.float32),
            torch.int8,
            None,
            "minval shape.*scale shape.*mismatch",
            id="int_minval_shape_mismatch",
        ),
        pytest.param(
            torch.randint(0, 10, (4,), dtype=torch.int8),
            torch.tensor(1.0, dtype=torch.float32),
            None,
            torch.tensor(0.0, dtype=torch.float16),
            torch.int8,
            None,
            "minval dtype.*must match scale dtype",
            id="int_minval_dtype_mismatch",
        ),
        pytest.param(
            torch.randint(0, 10, (4,), dtype=torch.int8),
            torch.tensor(1.0, dtype=torch.float32),
            None,
            torch.tensor(0.0, dtype=torch.float32),
            None,
            None,
            "input_dtype is required when minval is provided",
            id="int_minval_missing_input_dtype",
        ),
        pytest.param(
            torch.randint(0, 10, (4,), dtype=torch.int8),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(0, dtype=torch.int8),
            torch.tensor(0.0, dtype=torch.float32),
            torch.int8,
            None,
            "zero_point and minval are mutually exclusive",
            id="int_both_zp_and_minval",
        ),
    ],
)
def test_dequantize_validation(  # noqa: PLR0913
    input_t: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    minval: torch.Tensor | None,
    input_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
    match: str,
) -> None:
    """Validate that invalid inputs raise ValueError."""
    with pytest.raises(ValueError, match=match):
        dequantize(
            input_t,
            scale,
            zero_point=zero_point,
            minval=minval,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )


# ──────────────────────────────────────────────────────────────────────
# Roundtrip: quantize → dequantize
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("input_t", "scale", "zero_point", "axis"),
    [
        pytest.param(
            torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(0, dtype=torch.int8),
            0,
            id="per_tensor",
        ),
        pytest.param(
            torch.tensor(
                [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]],
                dtype=torch.float32,
            ),
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([0, 0], dtype=torch.int8),
            1,
            id="per_axis",
        ),
        pytest.param(
            torch.tensor(
                [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]],
                dtype=torch.float32,
            ),
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            torch.tensor([[0, 0]], dtype=torch.int8),
            1,
            id="per_axis_2d_scale",
        ),
    ],
)
def test_quantize_then_dequantize_roundtrip(
    input_t: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
) -> None:
    """Roundtrip: dequantize(quantize(x)) should recover x for exact multiples of scale."""
    quantized = quantize(
        input_t,
        scale,
        torch.int8,
        zero_point=zero_point,
        axis=axis,
    )
    assert quantized.dtype == torch.int8

    dequantized = dequantize(
        quantized,
        scale,
        zero_point=zero_point,
        axis=axis,
    )
    assert dequantized.dtype == scale.dtype

    torch.testing.assert_close(dequantized, input_t)


# ──────────────────────────────────────────────────────────────────────
# _inject_subbyte_in_quant — nbits inference
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("input_dtype", "expected_nbits"),
    [
        pytest.param(torch.int4, 4, id="int4"),
        pytest.param(torch.int8, 8, id="int8_noop"),
    ],
)
def test_inject_subbyte_uses_input_dtype_arg(
    input_dtype: torch.dtype,
    expected_nbits: int,
) -> None:
    """
    Verify _inject_subbyte_in_quant reads input_dtype from node.args (not kwargs).

    torch.export places all parameters — including defaults — into node.args as
    positional arguments.  This test uses data in range [-1, 1] which the
    heuristic would resolve to nbits=2, but input_dtype=int4 should override.
    """

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Data in [-1, 1]: heuristic would pick nbits=2, but input_dtype overrides.
            self.register_buffer(
                "quantized_data",
                torch.randint(-1, 2, (4, 8), dtype=torch.int8),
            )
            self.register_buffer("scale", torch.ones((2, 2), dtype=torch.float32))
            self.register_buffer("minval", torch.zeros((2, 2), dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dequant = torch.ops.coreai.constexpr_blockwise_shift_scale(
                self.quantized_data,
                self.scale,
                minval=self.minval,
                input_dtype=input_dtype,
            )
            return x + dequant

    model = Model().eval()
    with torch.no_grad():
        program = torch.export.export(model, (torch.randn(4, 8),))

    program = _inject_subbyte_in_quant(program)

    # Find the state_dict entry for quantized_data.
    data_key = next(k for k in program.state_dict if "quantized_data" in k)
    converted = program.state_dict[data_key]

    if expected_nbits < 8:
        assert isinstance(converted, SubbyteTensor), (
            f"Expected SubbyteTensor for {input_dtype}, got {type(converted)}"
        )
        assert converted.nbits == expected_nbits
    else:
        # nbits == 8 means no conversion (CHAR_BIT check skips it).
        assert not isinstance(converted, SubbyteTensor)


# ──────────────────────────────────────────────────────────────────────
# sparse_to_dense — validation tests
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("nonzero_data", "mask", "match"),
    [
        pytest.param(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([1, 0, 1, 0], dtype=torch.uint8),
            "nonzero_data must be a 1-D tensor",
            id="nonzero_data_not_1d",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([1, 0, 1, 0], dtype=torch.uint8),
            "nonzero_data dtype.*is not supported",
            id="unsupported_dtype",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([1, 0, 2, 0], dtype=torch.uint8),
            "mask must contain only 0 and 1 values",
            id="mask_non_binary",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            torch.tensor([1, 0, 1, 0], dtype=torch.uint8),
            "nonzero_data has 3 elements.*2 non-zero entries",
            id="count_mismatch",
        ),
    ],
)
def test_sparse_to_dense_validation(
    nonzero_data: torch.Tensor,
    mask: torch.Tensor,
    match: str,
) -> None:
    """Validate that invalid inputs raise ValueError."""
    with pytest.raises(ValueError, match=match):
        sparse_to_dense(nonzero_data, mask)
