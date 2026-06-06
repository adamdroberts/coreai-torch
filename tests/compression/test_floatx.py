# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test floatx tensors."""

import torch

from coreai_torch._compression._floatx import Float4Tensor


class TestBasics:
    """Test basic tensor properties."""

    @staticmethod
    def test_fp4_tensor_properties() -> None:
        """Test tensor shape and nbits properties."""
        elem = torch.tensor([0x00, 0x11], dtype=torch.uint8)
        fp4_tensor = Float4Tensor(elem)
        assert fp4_tensor.shape == torch.Size([4])
        assert fp4_tensor.nbits == 4

    @staticmethod
    def test_fp4_future_dtype() -> None:
        """future_dtype must be set to torch.float4_e2m1fn_x2."""
        elem = torch.tensor([0x00, 0x11], dtype=torch.uint8)
        fp4_tensor = Float4Tensor(elem)
        assert hasattr(fp4_tensor.elem, "future_dtype")
        assert fp4_tensor.elem.future_dtype is torch.float4_e2m1fn_x2

    @staticmethod
    def test_fp4_unpack() -> None:
        """Test FP4 unpacking with known values."""
        # Create packed uint8 elem with known FP4 values
        # Example: 2 FP4 values packed in 1 byte
        # 0x24 = 0b00100100 -> low=4 (2.0), high=2 (1.0)
        elem = torch.tensor([0x24], dtype=torch.uint8)
        fp4_tensor = Float4Tensor(elem)
        unpacked = fp4_tensor.unpack()
        # Verify unpacked values
        assert unpacked.shape == torch.Size([2])
        # Based on (low, high) order: [2.0, 1.0]
        assert unpacked[0].item() == 2.0
        assert unpacked[1].item() == 1.0


class TestOperation:
    """Test PyTorch tensor operations."""

    @staticmethod
    def test_fp4_slice() -> None:
        """Test slice."""
        elem = torch.randint(0, 256, size=(2, 4, 3), dtype=torch.uint8)
        fp4_tensor = Float4Tensor(elem)
        result = fp4_tensor[:, ::2, :]
        assert isinstance(result, Float4Tensor)
        assert result.shape == torch.Size([2, 2, 6])

    @staticmethod
    def test_fp4_contiguous() -> None:
        """Test contiguous."""
        elem = torch.randint(0, 256, size=(2, 4, 3), dtype=torch.uint8)
        fp4_tensor = Float4Tensor(elem)
        result = fp4_tensor.contiguous()
        assert isinstance(result, Float4Tensor)
        assert torch.all(fp4_tensor.elem == result.elem)

    @staticmethod
    def test_fp4_unsqueeze() -> None:
        """Test unsqueeze."""
        elem = torch.randint(0, 256, size=(2, 4, 3), dtype=torch.uint8)
        fp4_tensor = Float4Tensor(elem)
        result = fp4_tensor.unsqueeze(0)
        assert isinstance(result, Float4Tensor)
        assert result.shape == torch.Size([1, 2, 4, 6])
