# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Compression related PyTorch Module and custom ops."""

import math
from typing import cast

import torch
from packaging.version import Version
from typing_extensions import Self

import coreai_torch._compression.utils as compression_utils
from coreai_torch._compression._floatx import (
    Float4Tensor,
    byte_shape_to_fp4_shape,
    unpack_fp4,
)


@torch.library.custom_op("coreai::lut_to_dense", mutates_args=())
def lut_to_dense(
    indices: torch.Tensor,
    lut: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    """
    Define the custom pytorch coreai::lut_to_dense op. This operator is used to store constant weights in lookup tables format (aka palettized weights).

    Args:
    ----
    indices: The indices to look up LUT to derive the de-compressed output.
        * dtype IndicesT
        * If it is a tensor of torch.uint8, during lowering it will become uintx based on `NUM_PALETTES` in ``lut``.
          For example, if the ``lut`` has shape [2, 3, 16, 1], then the indices will be uint4.
    lut: The Look-up-table (LUT) to derive the de-compressed output.
        * shape [1.., NUM_PALETTES, VECTOR_SIZE]
            NUM_PALETTES is lut.shape[-2] and needs to be 2^nbits where nbits is indicated by IndicesT.
            VECTOR_SIZE is lut.shape[-1] and is added to support vector palettization. When VECTOR_SIZE is 1, it is scalar palettization.
        * lut's rank is K + 2, where K is the rank of indices.
            Each dimension of lut's first K dimensions should be divisible by each corresponding dimension of the decompressed tensor.
            e.g., when indices.shape = [2, 3, 4], lut.shape[:3] = [1, 1, 2], it means that there are two lookup tables over the 2nd axis.
            And each of them have their own lut values.
        * dtype T
    axis:
        * axis is used to define which axis the vectored elements in the lookup table be filled across the output tensor.
        * axis is only effective if VECTOR_SIZE is larger than 1.
        * dtype int16 (changed from int32 in the old operation)

    IndicesT: uint1, uint2, uint3, uint4, uint6, uint8
    T: uint8, int8, fp8_e5m2, fp8_e4m3fn, bf16, fp16, fp32

    Returns:
    -------
    De-palettized data which has the same dtype as `lut`. The output shape is:
        indices_shape * [1..1, VECTOR_SIZE, 1..1] (all 1 but VECTOR_SIZE at axis dimension).
    More specifically:
      output.shape[i] = indices.shape[i] , i != axis
      output.shape[i] = indices.shape[i] * VECTOR_SIZE, i == axis

    """
    num_palettes = lut.shape[-2]
    nbits = int(math.log2(num_palettes))
    if 2**nbits != num_palettes:
        err_msg = f"The shape of `lut` is invalid: the lut.shape[-2] should be equal to 2**nbits, but got {num_palettes}"
        raise ValueError(err_msg)

    if indices.dtype != torch.uint8:
        if not torch.all(indices == indices.to(torch.uint8)):
            err_msg = f"The `indices` has to be unsigned int data within uint8 range, but got {indices.dtype}."
            raise ValueError(err_msg)
        indices = indices.to(dtype=torch.uint8)

    if lut.dtype == torch.float64:
        lut = lut.to(torch.float32)
    supported_lut_dtype = (
        torch.uint8,
        torch.int8,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float16,
        torch.float32,
    )
    if lut.dtype not in supported_lut_dtype:
        err_msg = f"The `lut` dtype {lut.dtype} is not supported. Support dtypes: {supported_lut_dtype}"
        raise ValueError(err_msg)

    if axis < 0:
        axis += len(indices.shape)
    vector_size = lut.shape[-1]

    output_shape = list(indices.shape)
    output_shape[axis] *= vector_size

    flattened_indices = indices.flatten()
    flattened_repeated_lut = compression_utils.repeat_tensor_as(
        lut,
        indices.shape,
    ).reshape(
        len(flattened_indices),
        lut.shape[-2],
        lut.shape[-1],
    )

    # Cast float8 LUTs to float32 before advanced indexing, then cast back.
    # PyTorch < 2.10 does not support advanced indexing on float8 tensors.
    orig_dtype = flattened_repeated_lut.dtype
    needs_cast = orig_dtype in (torch.float8_e5m2, torch.float8_e4m3fn) and Version(
        torch.__version__
    ) < Version("2.10")
    if needs_cast:
        flattened_repeated_lut = flattened_repeated_lut.to(torch.float32)

    # Gather the palette entry for each index.
    # flattened_repeated_lut: [N, NUM_PALETTES, VECTOR_SIZE]
    # Result: [N, VECTOR_SIZE]
    flattened_output = flattened_repeated_lut[
        torch.arange(flattened_repeated_lut.size(0)),
        flattened_indices.long(),
    ]

    if needs_cast:
        flattened_output = flattened_output.to(orig_dtype)

    if vector_size == 1:
        return flattened_output.reshape(indices.shape)
    else:
        # [N, VECTOR_SIZE] → [*indices.shape, VECTOR_SIZE] → move VECTOR_SIZE next to axis → merge into axis
        intermediate = flattened_output.reshape(*indices.shape, vector_size)
        intermediate = torch.moveaxis(intermediate, -1, axis + 1)
        return intermediate.reshape(output_shape)


@torch.library.register_fake("coreai::lut_to_dense")  # type: ignore [misc]
def _(indices: torch.Tensor, lut: torch.Tensor, axis: int) -> torch.Tensor:
    output_shape = list(indices.shape)
    if axis < 0:
        axis += len(output_shape)
    output_shape[axis] *= lut.shape[-1]
    # Fake implementation with the right output shape.
    return torch.ones(output_shape, dtype=lut.dtype)


class PalettizeModule(torch.nn.Module):
    """Module to represent a palettized weight."""

    def __init__(
        self,
        indices: torch.Tensor,
        lut: torch.Tensor,
        vector_axis: int | None = None,
    ) -> None:
        """Palettization module using torch coreai::lut_to_dense op."""
        super().__init__()
        self.register_buffer("indices", indices)
        self.register_buffer("lut", lut)
        self.vector_axis = 0 if vector_axis is None else vector_axis

    def forward(self) -> torch.Tensor:
        """Forward function for PalettizeModule."""
        output = torch.ops.coreai.lut_to_dense(
            self.indices,
            self.lut,
            self.vector_axis,  # Note: still using vector_axis internally for compatibility
        )
        return cast("torch.Tensor", output)


def _expand_tensor(tensor: torch.Tensor, axis: int, rank: int) -> torch.Tensor:
    """Expand a 1d tensor to have rank of `rank` and dimension at `axis`."""
    target_shape = [1] * rank
    target_shape[axis] = torch.numel(tensor)
    return tensor.reshape(target_shape)


def is_fp4_quantization(data: torch.Tensor, scale: torch.Tensor) -> bool:
    """
    Check if quantization compresses data to float4.

    TODO: stuck in torch 2.7 that has no float4
    Remove this helper since can simply check data.dtype == torch.float4_e2m1fn_x2
    """
    return data.dtype == torch.uint8 and scale.dtype == torch.float8_e8m0fnu


def _validate_shift_scale(  # noqa: C901, PLR0912, PLR0913
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    minval: torch.Tensor | None,
    *,
    is_fp: bool,
    is_fp4: bool,
    input_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
) -> None:
    """Validate inputs for constexpr_blockwise_shift_scale."""
    # zero_point and minval are mutually exclusive
    if zero_point is not None and minval is not None:
        msg = "zero_point and minval are mutually exclusive; provide at most one."
        raise ValueError(msg)

    # Logical data shape accounts for fp4 packing (2 values per uint8 byte)
    effective_input_shape = (
        byte_shape_to_fp4_shape(input.shape) if is_fp4 else input.shape
    )

    # Scale and input must have same rank
    if len(scale.shape) != len(input.shape):
        msg = (
            f"scale rank {len(scale.shape)} and input rank {len(input.shape)} mismatch."
        )
        raise ValueError(msg)

    # Scale shape must divide input shape element-wise
    for d in range(len(effective_input_shape)):
        if effective_input_shape[d] % scale.shape[d] != 0:
            msg = (
                f"input shape {effective_input_shape} is not element-wise divisible"
                f" by scale shape {scale.shape} at dimension {d}."
            )
            raise ValueError(msg)

    if is_fp:
        # FP dequantization: output_dtype required when scale is e8m0fnu
        # (power-of-2 scale is not a valid output dtype, so it must be explicit).
        # For other FP scale types, output_dtype defaults to scale.dtype.
        if scale.dtype == torch.float8_e8m0fnu and output_dtype is None:
            msg = "output_dtype is required for FP dequantization with e8m0fnu scale."
            raise ValueError(msg)
        if zero_point is not None:
            msg = "zero_point is not supported for FP dequantization."
            raise ValueError(msg)
        if minval is not None:
            msg = "minval is not supported for FP dequantization."
            raise ValueError(msg)
    else:
        # INT dequantization
        if minval is not None:
            if input_dtype is None:
                msg = (
                    "input_dtype is required when minval is provided, because"
                    " input.dtype may not reflect the logical sub-byte type"
                    " (e.g., int4 is stored as int8) and input_dtype is needed"
                    " to compute q_min."
                )
                raise ValueError(msg)
            if minval.dtype != scale.dtype:
                msg = (
                    f"minval dtype {minval.dtype} must match scale dtype {scale.dtype}."
                )
                raise ValueError(msg)
            if minval.shape != scale.shape:
                msg = f"minval shape {minval.shape} and scale shape {scale.shape} mismatch."
                raise ValueError(msg)
        if zero_point is not None:
            if zero_point.dtype != input.dtype:
                msg = f"zero_point dtype {zero_point.dtype} must match input dtype {input.dtype}."
                raise ValueError(msg)
            if zero_point.shape != scale.shape:
                msg = f"zero_point shape {zero_point.shape} and scale shape {scale.shape} mismatch."
                raise ValueError(msg)


@torch.library.custom_op(
    "coreai::constexpr_blockwise_shift_scale",
    mutates_args=(),
)
def constexpr_blockwise_shift_scale(  # noqa: PLR0913
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    minval: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Define the custom pytorch coreai::constexpr_blockwise_shift_scale op.

    Blockwise dequantization of quantized weights. Same operation as
    coreai::dequantize but with blockwise scale (same rank as input)
    and FP4 support.

    Integer dequantization (zero_point mode):
        output = scale * (input - zero_point)
    Integer dequantization (minval mode):
        output = scale * (input - q_min) + minval
    Float dequantization:
        output = cast(input, output_dtype) * scale

    Arguments:
    ---------
    input: The quantized data tensor.
        * dtype SrcT.
    scale: The scale to use for dequantization.
        * Must have the same rank as ``input``.
        * Each dimension of scale must divide the corresponding dimension of input,
          i.e. ``input.shape[d] % scale.shape[d] == 0``.
        * dtype DstT for INT, power-of-2 float (e.g. float8_e8m0fnu) for FP.
    zero_point: Optional zero-point offset (mutually exclusive with minval).
        * Must have the same shape as ``scale``.
        * dtype SrcT (same as input).
    minval: Optional minimum-value offset (mutually exclusive with zero_point).
        * Must have the same shape as ``scale``.
        * dtype DstT (same as scale).
        * Requires ``input_dtype``.
    input_dtype: The logical dtype of quantized input. Required with ``minval``
        because input.dtype may not reflect the actual sub-byte type
        (e.g., int4 stored as int8), and input_dtype is needed to compute q_min.
    output_dtype: The desired output dtype. Required for FP dequantization
        when scale is float8_e8m0fnu; otherwise defaults to scale.dtype.

    SrcT: uint4, int4, uint8, int8, fp8_e5m2, fp8_e4m3fn, float4_e2m1fn (via uint8 packing)
    DstT: bf16, fp16, fp32

    Returns:
    -------
    The dequantized tensor with same shape as input (or unpacked shape for FP4).

    """
    # TODO: stuck in torch 2.7 that has no float4
    # should only need input.is_floating_point() once input.dtype == torch.float4_e2m1fn_x2
    is_fp4 = is_fp4_quantization(input, scale)
    is_fp = input.is_floating_point() or is_fp4

    _validate_shift_scale(
        input,
        scale,
        zero_point,
        minval,
        is_fp=is_fp,
        is_fp4=is_fp4,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )

    if is_fp:
        # FP dequantization: output = cast(input, output_dtype) * scale
        if is_fp4:
            input = unpack_fp4(input)
        f32_input = input.to(torch.float32)
        f32_scale = scale.to(torch.float32)
        output = f32_input * compression_utils.repeat_tensor_as(f32_scale, input.shape)
        result_dtype = output_dtype if output_dtype is not None else scale.dtype
        output = output.to(result_dtype)
    else:
        # INT dequantization
        expanded_scale = compression_utils.repeat_tensor_as(scale, input.shape)
        if zero_point is not None:
            # zero_point mode: output = scale * (input - zero_point)
            expanded_zp = compression_utils.repeat_tensor_as(zero_point, input.shape)
            output = (input.to(scale.dtype) - expanded_zp) * expanded_scale
        elif minval is not None:
            # minval mode: output = scale * (input - q_min) + minval
            assert input_dtype is not None  # enforced by validation
            q_min = compression_utils._int_dtype_min(input_dtype)
            expanded_minval = compression_utils.repeat_tensor_as(minval, input.shape)
            output = (input.to(scale.dtype) - q_min) * expanded_scale + expanded_minval
        else:
            # no-offset mode: output = scale * input
            output = input.to(scale.dtype) * expanded_scale
    return output


@torch.library.register_fake("coreai::constexpr_blockwise_shift_scale")  # type: ignore [misc]
def _(
    input: torch.Tensor,
    scale: torch.Tensor,
    _zero_point: torch.Tensor | None = None,
    _minval: torch.Tensor | None = None,
    _input_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # TODO: stuck in torch 2.7 that has no float4
    # should only need input.is_floating_point() once input.dtype == torch.float4_e2m1fn_x2
    is_fp_quantization = input.is_floating_point() or is_fp4_quantization(input, scale)
    if is_fp_quantization:
        if is_fp4_quantization(input, scale):
            output_shape = byte_shape_to_fp4_shape(input.shape)
        else:
            output_shape = input.shape
        result_dtype = output_dtype if output_dtype is not None else scale.dtype
    else:
        output_shape = input.shape
        result_dtype = scale.dtype
    # fake implementation with the right output shape
    return torch.ones(output_shape, dtype=result_dtype)


class ScaledPalettizeModule(torch.nn.Module):
    """Module to represent a scaled palettized weight."""

    def __init__(  # noqa: PLR0913
        self,
        indices: torch.Tensor,
        lut: torch.Tensor,
        scale: torch.Tensor,
        vector_axis: int | None = None,
        zero_point: torch.Tensor | None = None,
        minval: torch.Tensor | None = None,
        input_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize ScaledPalettizeModule with compression info registered in buffers."""
        super().__init__()
        self.register_buffer("indices", indices)
        self.register_buffer("lut", lut)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.register_buffer("minval", minval)
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.vector_axis = 0 if vector_axis is None else vector_axis

    def forward(self) -> torch.Tensor:
        """Forward function for ScaledPalettizeModule by calling torch custom coreai op."""
        dense = torch.ops.coreai.lut_to_dense(
            self.indices,
            self.lut,
            self.vector_axis,
        )
        output = torch.ops.coreai.constexpr_blockwise_shift_scale(
            dense,
            scale=self.scale,
            zero_point=self.zero_point,
            minval=self.minval,
            input_dtype=self.input_dtype,
            output_dtype=self.output_dtype,
        )
        return cast("torch.Tensor", output)


class WeightDequantizeModule(torch.nn.Module):
    """Module to represent a quantized weight using dequantization."""

    def __init__(  # noqa: PLR0913
        self: Self,
        quantized_data: torch.Tensor | Float4Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None = None,
        minval: torch.Tensor | None = None,
        input_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize WeightDequantizeModule with compression info registered in buffers."""
        super().__init__()
        self.register_buffer("quantized_data", quantized_data)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.register_buffer("minval", minval)
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def forward(self: Self) -> torch.Tensor:
        """Forward function for WeightDequantizeModule by calling torch custom coreai op."""
        output = torch.ops.coreai.constexpr_blockwise_shift_scale(
            self.quantized_data,
            self.scale,
            zero_point=self.zero_point,
            minval=self.minval,
            input_dtype=self.input_dtype,
            output_dtype=self.output_dtype,
        )
        return cast("torch.Tensor", output)


def _validate_quantize(  # noqa: C901, PLR0913
    input: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype,
    zero_point: torch.Tensor | None,
    minval: torch.Tensor | None,
    axis: int,
) -> None:
    """Validate inputs for coreai::quantize."""
    # zero_point and minval are mutually exclusive
    if zero_point is not None and minval is not None:
        msg = "zero_point and minval are mutually exclusive; provide at most one."
        raise ValueError(msg)

    # When scale is >1-D, its rank must match input rank
    if scale.ndim > 1 and scale.ndim != input.ndim:
        msg = (
            f"scale rank {scale.ndim} must match input rank {input.ndim}"
            f" when scale is not 0-D or 1-D."
        )
        raise ValueError(msg)

    # Scale must be reshapable to 0-D (per-tensor) or 1-D (per-channel)
    resolved_axis = axis if axis >= 0 else axis + input.ndim
    if scale.numel() != 1 and scale.numel() != input.shape[resolved_axis]:
        msg = (
            f"scale with {scale.numel()} elements cannot be reshaped to 0-D or 1-D"
            f" for input shape {tuple(input.shape)} along axis {axis}"
            f" (expected 1 or {input.shape[resolved_axis]} elements)."
        )
        raise ValueError(msg)

    is_fp = torch.zeros(1, dtype=output_dtype).is_floating_point()

    if is_fp:
        # FP quantization: offsets are not supported
        if zero_point is not None:
            msg = "zero_point is not supported for FP quantization."
            raise ValueError(msg)
        if minval is not None:
            msg = "minval is not supported for FP quantization."
            raise ValueError(msg)
    else:
        # INT quantization — note: zero_point dtype may not match output_dtype
        # due to torch sub-byte dtype limitations (e.g., int8 zero_point for int4 output)
        if zero_point is not None and zero_point.shape != scale.shape:
            msg = f"zero_point shape {zero_point.shape} and scale shape {scale.shape} mismatch."
            raise ValueError(msg)
        if minval is not None:
            if minval.dtype != input.dtype:
                msg = (
                    f"minval dtype {minval.dtype} must match input dtype {input.dtype}."
                )
                raise ValueError(msg)
            if minval.shape != scale.shape:
                msg = f"minval shape {minval.shape} and scale shape {scale.shape} mismatch."
                raise ValueError(msg)


@torch.library.custom_op("coreai::quantize", mutates_args=())
def quantize(  # noqa: PLR0913
    input: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype,
    zero_point: torch.Tensor | None = None,
    minval: torch.Tensor | None = None,
    axis: int = 0,
) -> torch.Tensor:
    """
    Define the custom pytorch coreai::quantize op.

    Supports both integer and floating-point quantization paths.
    This op does not support FP4 quantization.

    Integer quantization (zero_point mode):
        output = clip(round(input / scale) + zero_point, q_min, q_max)
    Integer quantization (minval mode):
        output = clip(round((input - minval) / scale) + q_min, q_min, q_max)
    Float quantization:
        output = cast(input / scale, output_dtype)

    Arguments:
    ---------
    input: The uncompressed input tensor.
        * dtype SrcT.
    scale: The scale to use for quantization.
        * ``scale.numel()`` must be 1 (per-tensor) or ``input.shape[axis]``
          (per-channel).  When rank > 1, must match input rank.
        * dtype SrcT.
    output_dtype: The desired output dtype. Always required — determines
        the quantized type (for INT: q_min/q_max clamp bounds; for FP:
        the target float type to cast to).
    zero_point: Optional zero-point offset (mutually exclusive with minval).
        * Must have the same shape as ``scale``.
        * dtype DstT (quantized domain).
    minval: Optional minimum-value offset (mutually exclusive with zero_point).
        * Must have the same shape as ``scale``.
        * dtype SrcT (same as input).
    axis: Only used if ``scale`` is a vector.
        * dtype int32.

    SrcT: bf16, fp16, fp32
    DstT: uint4, int4, uint8, int8, fp8_e5m2, fp8_e4m3fn

    Returns:
    -------
    The quantized tensor with same shape as input and dtype output_dtype.

    """
    is_fp = torch.zeros(1, dtype=output_dtype).is_floating_point()

    _validate_quantize(input, scale, output_dtype, zero_point, minval, axis)

    # Expand 0-D/1-D scale and offsets to input rank for broadcasting
    rank = len(input.shape)
    if scale.ndim in {0, 1}:
        scale = _expand_tensor(scale, axis, rank)
        if zero_point is not None:
            zero_point = _expand_tensor(zero_point, axis, rank)
        if minval is not None:
            minval = _expand_tensor(minval, axis, rank)

    if is_fp:
        # FP quantization: output = cast(clamp(input / scale, dtype_min, dtype_max), output_dtype)
        f32_input = input.to(torch.float32)
        f32_scale = scale.to(torch.float32)
        finfo = torch.finfo(output_dtype)
        output = (
            (f32_input / f32_scale).clamp(min=finfo.min, max=finfo.max).to(output_dtype)
        )
    else:
        # INT quantization
        q_min = compression_utils._int_dtype_min(output_dtype)
        q_max = compression_utils._int_dtype_max(output_dtype)
        if zero_point is not None:
            # zero_point mode: output = clip(round(input / scale) + zero_point, q_min, q_max)
            output = torch.clamp(
                torch.round(input / scale) + zero_point,
                q_min,
                q_max,
            ).to(output_dtype)
        elif minval is not None:
            # minval mode: output = clip(round((input - minval) / scale) + q_min, q_min, q_max)
            output = torch.clamp(
                torch.round((input - minval) / scale) + q_min,
                q_min,
                q_max,
            ).to(output_dtype)
        else:
            # no-offset mode: output = clip(round(input / scale), q_min, q_max)
            output = torch.clamp(
                torch.round(input / scale),
                q_min,
                q_max,
            ).to(output_dtype)

    return output


@torch.library.register_fake("coreai::quantize")  # type: ignore [misc]
def _(
    input: torch.Tensor,
    _scale: torch.Tensor,
    output_dtype: torch.dtype,
    _zero_point: torch.Tensor | None = None,
    _minval: torch.Tensor | None = None,
    _axis: int = 0,
) -> torch.Tensor:
    return torch.ones(list(input.shape), dtype=output_dtype)


def _validate_dequantize(  # noqa: C901, PLR0912, PLR0913
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    minval: torch.Tensor | None,
    input_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
    axis: int,
) -> None:
    """Validate inputs for coreai::dequantize."""
    is_fp = input.is_floating_point()

    # zero_point and minval are mutually exclusive
    if zero_point is not None and minval is not None:
        msg = "zero_point and minval are mutually exclusive; provide at most one."
        raise ValueError(msg)

    # When scale is >1-D, its rank must match input rank
    if scale.ndim > 1 and scale.ndim != input.ndim:
        msg = (
            f"scale rank {scale.ndim} must match input rank {input.ndim}"
            f" when scale is not 0-D or 1-D."
        )
        raise ValueError(msg)

    # Scale must be reshapable to 0-D (per-tensor) or 1-D (per-channel)
    resolved_axis = axis if axis >= 0 else axis + input.ndim
    if scale.numel() != 1 and scale.numel() != input.shape[resolved_axis]:
        msg = (
            f"scale with {scale.numel()} elements cannot be reshaped to 0-D or 1-D"
            f" for input shape {tuple(input.shape)} along axis {axis}"
            f" (expected 1 or {input.shape[resolved_axis]} elements)."
        )
        raise ValueError(msg)

    if is_fp:
        # FP dequantization: output_dtype required when scale is e8m0fnu
        if scale.dtype == torch.float8_e8m0fnu and output_dtype is None:
            msg = "output_dtype is required for FP dequantization with e8m0fnu scale."
            raise ValueError(msg)
        if zero_point is not None:
            msg = "zero_point is not supported for FP dequantization."
            raise ValueError(msg)
        if minval is not None:
            msg = "minval is not supported for FP dequantization."
            raise ValueError(msg)
    else:
        # INT dequantization
        if minval is not None:
            if input_dtype is None:
                msg = (
                    "input_dtype is required when minval is provided, because"
                    " input.dtype may not reflect the logical sub-byte type"
                    " (e.g., int4 is stored as int8) and input_dtype is needed"
                    " to compute q_min."
                )
                raise ValueError(msg)
            if minval.dtype != scale.dtype:
                msg = (
                    f"minval dtype {minval.dtype} must match scale dtype {scale.dtype}."
                )
                raise ValueError(msg)
            if minval.shape != scale.shape:
                msg = f"minval shape {minval.shape} and scale shape {scale.shape} mismatch."
                raise ValueError(msg)
        if zero_point is not None:
            if zero_point.dtype != input.dtype:
                msg = f"zero_point dtype {zero_point.dtype} must match input dtype {input.dtype}."
                raise ValueError(msg)
            if zero_point.shape != scale.shape:
                msg = f"zero_point shape {zero_point.shape} and scale shape {scale.shape} mismatch."
                raise ValueError(msg)


@torch.library.custom_op("coreai::dequantize", mutates_args=())
def dequantize(  # noqa: PLR0913
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    minval: torch.Tensor | None = None,
    axis: int = 0,
    input_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Define the custom pytorch coreai::dequantize op.

    Dequantization counterpart of coreai::quantize. Same operation as
    coreai::constexpr_blockwise_shift_scale but with per-tensor or
    per-axis scale (0-D or 1-D) and no FP4 support.

    Integer dequantization (zero_point mode):
        output = scale * (input - zero_point)
    Integer dequantization (minval mode):
        output = scale * (input - q_min) + minval
    Float dequantization:
        output = cast(input, output_dtype) * scale

    Arguments:
    ---------
    input: The quantized tensor to dequantize.
        * dtype SrcT.
    scale: The scale to use for dequantization.
        * ``scale.numel()`` must be 1 (per-tensor) or ``input.shape[axis]``
          (per-channel).  When rank > 1, must match input rank.
        * dtype DstT for INT, power-of-2 float (e.g. float8_e8m0fnu) for FP.
    zero_point: Optional zero-point offset (mutually exclusive with minval).
        * Must have the same shape as ``scale``.
        * dtype SrcT (same as input).
    minval: Optional minimum-value offset (mutually exclusive with zero_point).
        * Must have the same shape as ``scale``.
        * dtype DstT (same as scale).
        * Requires ``input_dtype``.
    axis: Only used if ``scale`` is a vector.
        * dtype int32.
    input_dtype: The logical dtype of quantized input. Required with ``minval``
        because input.dtype may not reflect the actual sub-byte type
        (e.g., int4 stored as int8), and input_dtype is needed to compute q_min.
    output_dtype: The desired output dtype. Required for FP dequantization
        when scale is float8_e8m0fnu; otherwise defaults to scale.dtype.

    SrcT: uint4, int4, uint8, int8, fp8_e5m2, fp8_e4m3fn
    DstT: bf16, fp16, fp32

    Returns:
    -------
    The dequantized tensor with same shape as input.

    """
    is_fp = input.is_floating_point()

    _validate_dequantize(
        input,
        scale,
        zero_point,
        minval,
        input_dtype,
        output_dtype,
        axis,
    )

    # Expand 0-D/1-D scale and offsets to input rank for broadcasting
    rank = len(input.shape)
    if scale.ndim in {0, 1}:
        scale = _expand_tensor(scale, axis, rank)
        if zero_point is not None:
            zero_point = _expand_tensor(zero_point, axis, rank)
        if minval is not None:
            minval = _expand_tensor(minval, axis, rank)

    if is_fp:
        # FP dequantization: output = cast(input, output_dtype) * scale
        result_dtype = output_dtype if output_dtype is not None else scale.dtype
        f32_input = input.to(torch.float32)
        f32_scale = scale.to(torch.float32)
        output = (f32_input * f32_scale).to(result_dtype)
    # INT dequantization
    elif zero_point is not None:
        # zero_point mode: output = scale * (input - zero_point)
        output = (input.to(scale.dtype) - zero_point) * scale
    elif minval is not None:
        # minval mode: output = scale * (input - q_min) + minval
        assert input_dtype is not None  # enforced by validation
        q_min = compression_utils._int_dtype_min(input_dtype)
        output = (input.to(scale.dtype) - q_min) * scale + minval
    else:
        # no-offset mode: output = scale * input
        output = input.to(scale.dtype) * scale

    return output


@torch.library.register_fake("coreai::dequantize")  # type: ignore [misc]
def _(
    input: torch.Tensor,
    scale: torch.Tensor,
    _zero_point: torch.Tensor | None = None,
    _minval: torch.Tensor | None = None,
    _axis: int = 0,
    _input_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    is_fp = input.is_floating_point()
    if is_fp:
        result_dtype = output_dtype if output_dtype is not None else scale.dtype
    else:
        result_dtype = scale.dtype
    return torch.ones(list(input.shape), dtype=result_dtype)


class ActivationQuantizeModule(torch.nn.Module):
    """Module to represent activation quantization."""

    def __init__(
        self,
        scale: torch.Tensor,
        output_dtype: torch.dtype,
        zero_point: torch.Tensor | None = None,
        minval: torch.Tensor | None = None,
        axis: int = 0,
    ) -> None:
        """Initialize ActivationQuantizeModule with compression info registered in buffers."""
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.register_buffer("minval", minval)
        self.output_dtype = output_dtype
        self.axis = axis

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward function for ActivationQuantizeModule by calling torch custom coreai op."""
        output = torch.ops.coreai.quantize(
            input_tensor,
            self.scale,
            self.output_dtype,
            zero_point=self.zero_point,
            minval=self.minval,
            axis=self.axis,
        )
        return cast("torch.Tensor", output)


class ActivationDequantizeModule(torch.nn.Module):
    """Module to represent activation dequantization."""

    def __init__(  # noqa: PLR0913
        self,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None = None,
        minval: torch.Tensor | None = None,
        axis: int = 0,
        input_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize ActivationDequantizeModule with compression info registered in buffers."""
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.register_buffer("minval", minval)
        self.axis = axis
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward function for ActivationDequantizeModule by calling torch custom coreai op."""
        output = torch.ops.coreai.dequantize(
            input_tensor,
            self.scale,
            zero_point=self.zero_point,
            minval=self.minval,
            axis=self.axis,
            input_dtype=self.input_dtype,
            output_dtype=self.output_dtype,
        )
        return cast("torch.Tensor", output)


def _validate_sparse_to_dense(
    nonzero_data: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Validate inputs for coreai::sparse_to_dense."""
    if nonzero_data.ndim != 1:
        msg = f"nonzero_data must be a 1-D tensor, but got {nonzero_data.ndim}-D."
        raise ValueError(msg)

    supported_dtype = (
        torch.uint8,
        torch.int8,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float16,
        torch.float32,
    )
    if nonzero_data.dtype not in supported_dtype:
        msg = (
            f"nonzero_data dtype {nonzero_data.dtype} is not supported."
            f" Supported dtypes: {supported_dtype}"
        )
        raise ValueError(msg)

    bool_mask = mask.bool()
    if not torch.equal(mask, bool_mask.to(mask.dtype)):
        msg = "mask must contain only 0 and 1 values."
        raise ValueError(msg)

    num_nonzero = bool_mask.sum().item()
    if nonzero_data.numel() != num_nonzero:
        msg = (
            f"nonzero_data has {nonzero_data.numel()} elements, but mask has"
            f" {num_nonzero} non-zero entries. They must match."
        )
        raise ValueError(msg)


@torch.library.custom_op("coreai::sparse_to_dense", mutates_args=())
def sparse_to_dense(
    nonzero_data: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Define the custom pytorch coreai::sparse_to_dense op. This operator is used to store constant weights in sparsified format.

    Args:
    ----
    nonzero_data: The non-zero entries in the weight.
        * Must be a 1-D tensor.
        * dtype T
    mask: The mask to indicate if an entry is zero or non-zero.
        * The mask uses 0 or 1 to indicate if the element at the corresponding index is zero or not.
          If the mask is 1, the corresponding element in the output tensor is non-zero and the value is from the ``nonzero_data``.
          Likewise, if the mask is 0, the corresponding element in the output tensor is zero.
        * dtype bool (or other dtypes such as uint8 which can be converted to bool)
        * During lowering it will become ui1 (unsigned 1-bit).

    T: uint8, int8, fp8_e5m2, fp8_e4m3fn, bf16, fp16, fp32

    Returns:
    -------
    The dense data which has the same shape as `mask`, with non-zero entries filled by `nonzero_data`.

    """
    _validate_sparse_to_dense(nonzero_data, mask)
    decompressed_val = torch.zeros_like(mask, dtype=nonzero_data.dtype)
    decompressed_val[mask] = nonzero_data
    return decompressed_val


@torch.library.register_fake("coreai::sparse_to_dense")  # type: ignore [misc]
def _(nonzero_data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Fake implementation with the right output shape.
    return torch.ones_like(mask, dtype=nonzero_data.dtype)


class SparseModule(torch.nn.Module):
    """Module to represent a sparse weight."""

    def __init__(
        self,
        nonzero_data: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Sparse module using torch coreai::sparse_to_dense op."""
        super().__init__()
        self.register_buffer("nonzero_data", nonzero_data)
        self.register_buffer("mask", mask)

    def forward(self) -> torch.Tensor:
        """Forward function for SparseModule by calling torch custom coreai op."""
        output = torch.ops.coreai.sparse_to_dense(
            self.nonzero_data,
            self.mask,
        )
        return cast("torch.Tensor", output)
