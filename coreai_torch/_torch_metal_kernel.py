# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Torch-integrated wrapper around Core AI's CustomMetalKernel."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Sequence
from functools import wraps
from typing import Any, Callable, get_args, get_origin

import torch
from coreai.authoring import CustomMetalKernel, MetalParameter
from torch._library.custom_ops import CustomOpDef
from typing_extensions import Self

# We're allowing for int, bool, and float scalar inputs.
_ALLOWED_SCALARS = {int, float, bool}

# Threads-per-grid / threads-per-threadgroup must be 3-tuples per the Metal
# `dispatchThreads` API.
_THREAD_TUPLE_LEN = 3


class TorchMetalKernel(CustomMetalKernel):
    """A :class:`CustomMetalKernel` that also registers a ``torch.library`` custom op.

    This subclass adds the PyTorch integration layer on top of the base
    class's kernel construction:

    * Validates the torch callable's input and return annotations.
    * Registers a ``torch.library.custom_op`` under the
      ``coreai_metal_kernels`` namespace so ``torch.export`` can trace calls.
    * Provides a ``__call__`` method that converts thread dispatch tuples
      to tensors and invokes the custom op.
    """

    torch_custom_op: CustomOpDef

    def __init__(  # noqa: PLR0913
        self: Self,
        name: str,
        input_names: list[str],
        result_names: list[str],
        src: str,
        torch_defn: Callable[..., Any],
        metal_params: list[MetalParameter] | None = None,
        helper_src: str | None = None,
        template_dtypes: dict[str, str] | None = None,
    ) -> None:
        """Construct a torch-integrated custom metal kernel.

        Args:
            name: Kernel identifier.
            input_names: Names matching the Metal source input variables.
            result_names: Names matching the Metal source output variables.
            src: Metal kernel body (signature is generated automatically).
            torch_defn: Reference PyTorch implementation for shape inference.
            metal_params: Metal thread attributes (e.g. ``thread_position_in_grid``).
            helper_src: Additional Metal helper functions.
            template_dtypes: Map of input names to dtype placeholder strings.
        """
        # Stash fields needed by validation and torch op construction
        # before super().__init__ runs.
        self.name = name
        self.input_names = input_names
        self.result_names = result_names

        self._validate_name(name)
        self._validate_io_names(input_names, result_names)

        # eval_str=True resolves PEP 563 string annotations introduced by
        # ``from __future__ import annotations`` in the caller's module. Without
        # this, ``param.annotation`` is the bare string "torch.Tensor" and the
        # identity checks in :meth:`_validate_torch_inputs` fail.
        torch_sig = inspect.signature(torch_defn, eval_str=True)
        self._validate_torch_inputs(torch_sig)
        self._validate_torch_returns(torch_sig)
        self.torch_custom_op = self._construct_torch_custom_op(torch_defn)

        super().__init__(
            name,
            input_names=input_names,
            result_names=result_names,
            src=src,
            metal_params=metal_params,
            helper_src=helper_src,
            template_dtypes=template_dtypes,
        )

    # ------------------------------------------------------------------
    # Torch validation
    # ------------------------------------------------------------------

    @property
    def result_shape_params(self: Self) -> list[str]:
        """Parameter names for per-result shape tensors."""
        return [f"result_shape_{name}" for name in self.result_names]

    @staticmethod
    def _validate_name(name: str) -> None:
        """Reject empty / whitespace-only kernel names.

        The Swift runtime (``CustomMetalKernel.swift``) treats an empty
        ``kernelName`` as a user error; catch it eagerly with a clear message
        rather than letting it surface deep in the runtime.
        """
        if not isinstance(name, str) or not name.strip():
            err = f"Kernel name must be a non-empty string, got {name!r}"
            raise ValueError(err)

    @staticmethod
    def _validate_io_names(
        input_names: list[str],
        result_names: list[str],
    ) -> None:
        """Reject empty / duplicated / overlapping input and result names.

        Duplicate or overlapping names produce ill-formed Metal kernel sources
        (two parameters with the same identifier) and confusing failures.
        """
        if not result_names:
            err = "result_names must contain at least one entry"
            raise ValueError(err)

        for label, names in (("input", input_names), ("result", result_names)):
            duplicates = sorted(n for n, c in Counter(names).items() if c > 1)
            if duplicates:
                err = f"Duplicate {label} names: {duplicates}"
                raise ValueError(err)

        overlap = sorted(set(input_names) & set(result_names))
        if overlap:
            err = f"Names appear in both input_names and result_names: {overlap}"
            raise ValueError(err)

    def _validate_torch_inputs(self: Self, torch_sig: inspect.Signature) -> None:
        """Ensure every parameter is torch.Tensor or an allowed scalar type."""
        for param in torch_sig.parameters.values():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                err = (
                    "custom kernels do not support variadic parameters "
                    f"(*args / **kwargs); got parameter '{param.name}' with kind "
                    f"{param.kind.description}"
                )
                raise TypeError(err)
            if param.annotation is torch.Tensor or param.annotation in _ALLOWED_SCALARS:
                continue
            err = (
                "custom kernels only support `torch.Tensor`, `float`, `bool` "
                f"and `int` inputs, got {param.annotation}"
            )
            raise TypeError(err)

        if len(torch_sig.parameters) != len(self.input_names):
            err = (
                "torch function should have same number of parameters as specified "
                f"by input names, expected {len(self.input_names)}, "
                f"got {len(torch_sig.parameters)}"
            )
            raise ValueError(err)

    def _validate_torch_returns(self: Self, torch_sig: inspect.Signature) -> None:
        """Ensure the return annotation is Tensor, list[Tensor], or tuple[Tensor, ...]."""
        annotation = torch_sig.return_annotation

        def _raise() -> None:
            err = (
                "Metal kernels only support return types of `torch.Tensor`, "
                "`list[torch.Tensor]`, or `tuple[torch.Tensor]` (with a concrete "
                "number of tuple members). The torch callback has a return type "
                f"of {annotation!s}"
            )
            raise TypeError(err)

        # Single-tensor return: exactly one result expected.
        if annotation is torch.Tensor:
            if len(self.result_names) != 1:
                err = (
                    "torch_defn returns a single torch.Tensor, but result_names "
                    f"has {len(self.result_names)} entries: {self.result_names}"
                )
                raise ValueError(err)
            return

        origin = get_origin(annotation)
        if not origin or origin not in {tuple, list}:
            _raise()

        annotation_args = get_args(annotation)
        for arg in annotation_args:
            if arg is not torch.Tensor:
                _raise()

        # `tuple[torch.Tensor, torch.Tensor]` has a concrete count we can enforce;
        # `list[torch.Tensor]` is variable-length, so we can only validate at call
        # time.
        if origin is tuple and len(annotation_args) != len(self.result_names):
            err = (
                f"torch_defn returns tuple of {len(annotation_args)} tensors, "
                f"but result_names has {len(self.result_names)} entries: "
                f"{self.result_names}"
            )
            raise ValueError(err)

    # ------------------------------------------------------------------
    # Torch custom op
    # ------------------------------------------------------------------

    def _construct_torch_custom_op(
        self: Self,
        torch_callable: Callable[..., Any],
    ) -> CustomOpDef:
        """Register a ``torch.library.custom_op`` from the provided callable."""
        # Resolve PEP 563 string annotations (see ``__init__`` for context).
        sig = inspect.signature(torch_callable, eval_str=True)

        # Augment the signature with thread dispatch and result shape params.
        extra_params = [
            inspect.Parameter(
                "threads_per_grid",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
            ),
            inspect.Parameter(
                "threads_per_thread_group",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
            ),
            *(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=torch.Tensor,
                )
                for name in self.result_shape_params
            ),
        ]
        augmented_sig = sig.replace(
            parameters=[*sig.parameters.values(), *extra_params],
        )

        original_param_count = len(sig.parameters)

        @wraps(torch_callable)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in {"threads_per_grid", "threads_per_thread_group"}
                and k not in self.result_shape_params
            }
            return torch_callable(*args[:original_param_count], **kwargs)

        wrapper.__signature__ = augmented_sig  # type: ignore[attr-defined]

        torch_custom_op = torch.library.custom_op(
            f"coreai_metal_kernels::{self.name}",
            mutates_args=(),
        )(wrapper)

        @torch_custom_op.register_fake
        def _(*args: Any) -> Any:
            res = wrapper(*args)
            if isinstance(res, Sequence):
                empty = [torch.empty_like(r) for r in res]
                return tuple(empty) if isinstance(res, tuple) else empty
            return res

        return torch_custom_op

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(
        self: Self,
        *args: Any,
        threads_per_grid: tuple[int, int, int],
        threads_per_thread_group: tuple[int, int, int],
        result_shapes: list[list[int]],
    ) -> Any:
        """Invoke the underlying torch custom op."""
        if len(threads_per_grid) != _THREAD_TUPLE_LEN:
            err = (
                f"threads_per_grid must be a 3-tuple, got {len(threads_per_grid)} "
                f"elements: {threads_per_grid!r}"
            )
            raise ValueError(err)
        if len(threads_per_thread_group) != _THREAD_TUPLE_LEN:
            err = (
                "threads_per_thread_group must be a 3-tuple, got "
                f"{len(threads_per_thread_group)} elements: "
                f"{threads_per_thread_group!r}"
            )
            raise ValueError(err)
        if len(result_shapes) != len(self.result_names):
            err = (
                f"result_shapes must contain one shape per result name; "
                f"expected {len(self.result_names)} (for {self.result_names}), "
                f"got {len(result_shapes)}"
            )
            raise ValueError(err)

        grid_tn = torch.tensor(list(threads_per_grid), dtype=torch.uint32)
        tgroup_tn = torch.tensor(list(threads_per_thread_group), dtype=torch.uint32)
        shape_tns = [torch.tensor(shape, dtype=torch.uint32) for shape in result_shapes]
        return self.torch_custom_op(*args, grid_tn, tgroup_tn, *shape_tns)
