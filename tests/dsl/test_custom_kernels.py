# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for custom metal kernels."""

import re
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from coreai_torch import (
    MetalParameter,
    TorchConverter,
    TorchMetalKernel,
    get_decomp_table,
)

from ..utils import filecheck_pattern


@pytest.fixture
def result_shape() -> list[int]:
    """Return type for result shape."""
    return [2, 2, 3]


@pytest.fixture
def custom_add() -> TorchMetalKernel:
    """Fixture for elementwise-add custom kernel."""

    def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    return TorchMetalKernel(
        "custom_add",
        input_names=["x", "y"],
        result_names=["sum"],
        src="sum[id] = x[id] + y[id];",
        torch_defn=torch_add,
        metal_params=[
            MetalParameter("id", "uint", "thread_position_in_grid"),
        ],
    )


@pytest.fixture
def metal_constraint(result_shape: list[int]) -> str:
    """Fixture for metal constraint string."""
    alignments = "x".join(["1"] * (len(result_shape) + 1))
    interleave = "x".join(["1"] * len(result_shape))
    return f"#coreaix.hw_constraints<MTLBuffer, alignments: [{alignments}], interleave: [{interleave}]>"


@pytest.fixture
def eval_model(custom_add: TorchMetalKernel, result_shape: list[int]) -> Any:
    """Fixture for evaluated model with custom kernel."""

    class MetalModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            custom_sum = custom_add(
                x,
                y,
                threads_per_grid=(12, 1, 1),
                threads_per_thread_group=(1, 1, 1),
                result_shapes=[result_shape],
            )
            return custom_sum

    return MetalModel().eval()


def _convert_model(
    model: Any,
    args: tuple,
    kernels: list[TorchMetalKernel],
    output_names: list[str] | None = None,
    dynamic_shapes: dict | None = None,
) -> Any:
    """Export, register, and convert a model with custom kernels."""
    exported = torch.export.export(model, args=args, dynamic_shapes=dynamic_shapes)
    ep = exported.run_decompositions(get_decomp_table())
    converter = TorchConverter()
    converter.register_custom_kernels(kernels)
    converter.add_exported_program(ep, output_names=output_names or [])
    return converter.to_coreai()


class TestMetalKernel:
    """Test that we can author custom metal kernels from Python and have them show up in the converted program."""

    @pytest.mark.ir
    @staticmethod
    def test_define_and_import_metal_kernel(
        custom_add: TorchMetalKernel,
        result_shape: list[int],
        metal_constraint: str,
        eval_model: Any,
    ) -> None:
        """We should be able to define and call a custom metal kernel."""
        coreai_program = _convert_model(
            eval_model,
            args=(
                torch.rand(*result_shape, dtype=torch.float16),
                torch.ones(*result_shape, dtype=torch.float16),
            ),
            kernels=[custom_add],
            output_names=["sum"],
        )

        ir_shape = "x".join([str(dim) for dim in result_shape])

        metal4_kernel_def = (
            "coreai.metal4_kernel "
            "kernel_args(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}), "
            "threads_per_grid %{{[a-z0-9_]+}}, "
            "threads_per_thread_group %{{[a-z0-9_]+}}, "
            "result_shapes(%{{[a-z0-9_]+}}) "
            '{kernel_name = "custom_add_{{[a-z]+}}", kernel_source = "{{.+}}"} : '
            f"(tensor<{ir_shape}xf16, {metal_constraint}>, "
            f"tensor<{ir_shape}xf16, {metal_constraint}>, "
            "tensor<3xui32>, "
            "tensor<3xui32>, "
            f"tensor<3xui32>) -> tensor<{ir_shape}xf16, {metal_constraint}>"
        )
        filecheck_pattern(
            str(coreai_program),
            f"""
            // CHECK: {metal4_kernel_def}
            """,
        )

    @pytest.mark.ir
    @staticmethod
    def test_import_dynamic_shape(
        custom_add: TorchMetalKernel,
        result_shape: list[int],
        metal_constraint: str,
        eval_model: Any,
    ) -> None:
        """We should be able to import dynamic-shape models with custom kernels."""
        dim = torch.export.Dim("dim", min=1, max=32)
        coreai_program = _convert_model(
            eval_model,
            args=(
                torch.rand(*result_shape, dtype=torch.float16),
                torch.ones(*result_shape, dtype=torch.float16),
            ),
            kernels=[custom_add],
            output_names=["sum"],
            dynamic_shapes={
                "x": {0: dim},
                "y": {0: dim},
            },
        )

        ir_shape = "x".join(
            [str(d) if idx != 0 else "?" for idx, d in enumerate(result_shape)],
        )

        metal4_kernel_def = (
            "coreai.metal4_kernel "
            "kernel_args(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}), "
            "threads_per_grid %{{[a-z0-9_]+}}, "
            "threads_per_thread_group %{{[a-z0-9_]+}}, "
            "result_shapes(%{{[a-z0-9_]+}}) "
            '{kernel_name = "custom_add_{{[a-z]+}}", kernel_source = "{{.+}}"} : '
            f"(tensor<{ir_shape}xf16, {metal_constraint}>, "
            f"tensor<{ir_shape}xf16, {metal_constraint}>, "
            "tensor<3xui32>, "
            "tensor<3xui32>, "
            f"tensor<3xui32>) -> tensor<{ir_shape}xf16, {metal_constraint}>"
        )
        filecheck_pattern(
            str(coreai_program),
            f"""
            // CHECK: {metal4_kernel_def}
            """,
        )

    @staticmethod
    def test_too_many_parameters() -> None:
        """We should not allow for users to specify kernels with more parameters than metal allows."""

        def torch_sum(  # noqa: PLR0913
            arg_0: torch.Tensor,
            arg_1: torch.Tensor,
            arg_2: torch.Tensor,
            arg_3: torch.Tensor,
            arg_4: torch.Tensor,
            arg_5: torch.Tensor,
            arg_6: torch.Tensor,
            arg_7: torch.Tensor,
            arg_8: torch.Tensor,
            arg_9: torch.Tensor,
        ) -> torch.Tensor:
            return (
                arg_0
                + arg_1
                + arg_2
                + arg_3
                + arg_4
                + arg_5
                + arg_6
                + arg_7
                + arg_8
                + arg_9
            )

        # Metal 4: params = inputs + results + metal_params. Use 21 metal params so
        # 10 inputs + 1 result + 21 metal_params = 32 > PARAMETER_LIMIT (31).
        kernel = TorchMetalKernel(
            "custom_sum",
            input_names=[f"arg_{idx}" for idx in range(10)],
            result_names=["result"],
            src="...",
            torch_defn=torch_sum,
            metal_params=[
                MetalParameter(f"extra_{idx}", "uint", "thread_position_in_grid")
                for idx in range(21)
            ],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    threads_per_grid=(1, 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        model = Model().eval()
        err = "metal kernels support 31 inputs, got 32"
        with pytest.raises(ValueError, match=err):
            _convert_model(
                model,
                args=(torch.rand(2, 2, 4, dtype=torch.float32),),
                kernels=[kernel],
            )

    @staticmethod
    def test_torch_param_number_does_not_match_kernel() -> None:
        """The number of input names should match what is specified by the torch op."""

        def torch_single(x: torch.Tensor) -> torch.Tensor:
            return x

        err = (
            "torch function should have same number of parameters as specified by input "
            "names, expected 2, got 1"
        )
        with pytest.raises(ValueError, match=err):
            TorchMetalKernel(
                "custom_kernel",
                input_names=["x", "y"],
                result_names=["result"],
                src="...",
                torch_defn=torch_single,
            )

    @staticmethod
    def test_reject_unsupported_input_types() -> None:
        """Users can specify functions that take in bools, ints, floats, and tensors."""

        def torch_str(x: torch.Tensor, s: str) -> torch.Tensor:  # noqa: ARG001
            return x

        err = (
            "custom kernels only support `torch.Tensor`, `float`, `bool` and `int` inputs, "
            "got <class 'str'>"
        )
        with pytest.raises(TypeError, match=err):
            TorchMetalKernel(
                "custom_kernel",
                input_names=["x", "s"],
                result_names=["result"],
                src="...",
                torch_defn=torch_str,
            )

    @staticmethod
    def test_validate_torch_return() -> None:
        """Users can only provide torch functions that return Tensor, list[Tensor], or tuple[Tensor]."""

        def construct_err(t: Any) -> str:
            return re.escape(
                "Metal kernels only support return types of `torch.Tensor`, "
                "`list[torch.Tensor]`, or `tuple[torch.Tensor]` (with a concrete "
                "number of tuple members). The torch callback has a return type "
                f"of {t!s}",
            )

        def torch_arbitrary_tuple(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            return x, y

        def torch_int(x: torch.Tensor) -> int:
            return x.numel()

        def torch_seq(x: torch.Tensor, y: torch.Tensor) -> Sequence[torch.Tensor]:
            return [x, y]

        with pytest.raises(TypeError, match=construct_err(tuple[torch.Tensor, ...])):
            TorchMetalKernel(
                "arbitrary",
                input_names=["x", "y"],
                result_names=["result"],
                src="...",
                torch_defn=torch_arbitrary_tuple,
            )

        with pytest.raises(TypeError, match=construct_err(int)):
            TorchMetalKernel(
                "int",
                input_names=["x"],
                result_names=["result"],
                src="...",
                torch_defn=torch_int,
            )

        with pytest.raises(TypeError, match=construct_err(Sequence[torch.Tensor])):
            TorchMetalKernel(
                "seq",
                input_names=["x", "y"],
                result_names=["result"],
                src="...",
                torch_defn=torch_seq,
            )

    @staticmethod
    def test_dtype_template_not_in_inputs() -> None:
        """We should crash if we provide a dtype template that maps to a nonexistent input."""

        def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        with pytest.raises(ValueError, match=re.escape("Inputs {'z'} not specified")):
            TorchMetalKernel(
                "missing_input",
                input_names=["x", "y"],
                result_names=["result"],
                src="...",
                torch_defn=torch_add,
                template_dtypes={"z": "TYPE"},
            )

    @staticmethod
    def test_duplicate_dtype_templates() -> None:
        """We should crash if we provide duplicate dtype templates."""

        def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        with pytest.raises(
            ValueError,
            match=re.escape("Provided duplicated template strings ['TYPE']"),
        ):
            TorchMetalKernel(
                "missing_input",
                input_names=["x", "y"],
                result_names=["result"],
                src="...",
                torch_defn=torch_add,
                template_dtypes={"x": "TYPE", "y": "TYPE"},
            )

    @staticmethod
    def test_unsupported_input_dtype() -> None:
        """Kernel inputs with dtypes not in metal_type_mappings raise TypeError during conversion."""

        def torch_fn(x: torch.Tensor) -> torch.Tensor:
            return x

        kernel = TorchMetalKernel(
            "f8_input_kernel",
            input_names=["x"],
            result_names=["result"],
            src="result[id] = x[id];",
            torch_defn=torch_fn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(1, 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        model = Model().eval()
        with pytest.raises(
            TypeError,
            match="kernel input at index 0 has unsupported dtype: f8E5M2",
        ):
            _convert_model(
                model,
                args=(torch.zeros(4, dtype=torch.float8_e5m2),),
                kernels=[kernel],
            )

    @staticmethod
    def test_unsupported_result_dtype() -> None:
        """Kernel results with dtypes not in metal_type_mappings raise TypeError during conversion."""

        def torch_fn(x: torch.Tensor) -> torch.Tensor:
            return x.to(torch.float8_e5m2)

        kernel = TorchMetalKernel(
            "f8_result_kernel",
            input_names=["x"],
            result_names=["result"],
            src="result[id] = x[id];",
            torch_defn=torch_fn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(1, 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        model = Model().eval()
        with pytest.raises(
            TypeError,
            match="Result type at index 0 has unsupported dtype: f8E5M2",
        ):
            _convert_model(
                model,
                args=(torch.rand(4, dtype=torch.float32),),
                kernels=[kernel],
            )
