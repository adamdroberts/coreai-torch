# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Validate that subgraph patterns of elementwise ops are correctly imported and lowered."""

import pytest
import torch
from torch import nn

from ..utils import validate_numerical_output


@pytest.mark.parametrize(
    "a,b",  # noqa: PT006
    [
        (torch.rand(1, 2048, 1, 2), torch.rand(2).to(torch.float16)),
        (
            torch.randint(low=1, high=100, size=(3, 3), dtype=torch.int32),
            torch.randint(low=1, high=100, size=(3, 3), dtype=torch.int32),
        ),
        (
            torch.randint(low=1, high=100, size=(3, 3), dtype=torch.int16),
            torch.randint(low=1, high=100, size=(3, 3), dtype=torch.int16),
        ),
        (
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),
            torch.tensor([-1.5], dtype=torch.float32),
        ),
    ],
)
async def test_long_arithmetic_chain_add_sub(a: torch.Tensor, b: torch.Tensor) -> None:
    """Test for long sequence of add / sub with a constant."""

    class Model(nn.Module):
        def __init__(self, is_float: bool, shape: list[int]):
            super().__init__()
            if is_float:
                self.a_modifier = torch.rand(*(shape))
                self.b_modifier = torch.rand(*(shape))
            else:
                self.a_modifier = torch.randint(-1000, 1000, (shape))
                self.b_modifier = torch.randint(-1000, 1000, (shape))

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            for i in range(100):
                if i & 1:
                    a = a + self.a_modifier
                    b = b - self.b_modifier
                else:
                    a = a - self.a_modifier
                    b = b + self.b_modifier
            return a * b

    inputs = {
        "model": Model(is_float=a.dtype.is_floating_point, shape=a.shape),
        "a": a,
        "b": b,
    }
    await validate_numerical_output(**inputs)


@pytest.mark.parametrize(
    "a,b,num_iters",  # noqa: PT006
    [
        (torch.rand(1, 2048, 1, 2), torch.rand(2).to(torch.float16), 10),
        (
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),
            torch.tensor([-1.5], dtype=torch.float32),
            10,
        ),
    ],
)
async def test_long_arithmetic_chain_mul_divide(
    a: torch.Tensor,
    b: torch.Tensor,
    num_iters: int,
) -> None:
    """Test for long chain of mul / divide ops with a constant."""

    class Model(nn.Module):
        def __init__(self, dtype: "torch.dtype", shape: list[int], num_iters: int):
            super().__init__()
            self.num_iters = num_iters
            if dtype.is_floating_point:
                self.a_modifier = torch.rand(*(shape), dtype=dtype)
                self.b_modifier = torch.rand(*(shape), dtype=dtype)
            else:
                self.a_modifier = torch.randint(-10, 10, (shape), dtype=dtype)
                self.b_modifier = torch.randint(-10, 10, (shape), dtype=dtype)
            self.const_three = torch.tensor(3).to(dtype)
            self.const_two = torch.tensor(2).to(dtype)

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            for i in range(self.num_iters):
                if i & 1:
                    a = a * self.a_modifier
                    b = b / self.const_three
                else:
                    a = a / self.const_two
                    b = b * self.b_modifier
            return a + b

    inputs = {
        "model": Model(
            dtype=a.dtype,
            shape=a.shape,
            num_iters=num_iters,
        ),
        "a": a,
        "b": b,
    }
    await validate_numerical_output(**inputs)


@pytest.mark.parametrize(
    "a,b,num_iters",  # noqa: PT006
    [
        (
            torch.randint(low=1, high=100, size=(3, 3), dtype=torch.int32),
            torch.randint(low=1, high=100, size=(3, 3), dtype=torch.int32),
            10,
        ),
        (
            torch.randint(low=1, high=5, size=(3, 3), dtype=torch.int16),
            torch.randint(low=1, high=5, size=(3, 3), dtype=torch.int16),
            10,
        ),
    ],
)
async def test_long_arithmetic_chain_mul(
    a: torch.Tensor,
    b: torch.Tensor,
    num_iters: int,
) -> None:
    """Test for long chain of mul ops for ints."""

    class Model(nn.Module):
        def __init__(self, num_iters: int):
            super().__init__()
            self.num_iters = num_iters
            self.a_modifier = torch.tensor(4)
            self.b_modifier = torch.tensor(3)

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            am = self.a_modifier.to(a.dtype)
            bm = self.b_modifier.to(a.dtype)
            for _ in range(self.num_iters):
                a = a * am
                b = b * bm
            return a + b

    inputs = {
        "model": Model(
            num_iters=num_iters,
        ),
        "a": a,
        "b": b,
    }
    await validate_numerical_output(**inputs)
