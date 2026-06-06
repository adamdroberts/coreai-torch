# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union, cast

import coreai
import numpy as np
import torch
from numpy.typing import NDArray

from .debug_info import DebugInfoRecord, parse_debug_infos
from .torch_utils import _TorchFXNodeValueInterpreter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MappingKey:
    """Key for mapping ODIX outputs to source outputs."""

    odix_id: int
    delegate_id: int | None
    output_idx: int


@dataclass(frozen=True)
class _CompiledIdMappings:
    """Result of building compiled ID mappings."""

    target_to_source_output_map: dict[_MappingKey, tuple[int, int]]
    all_compiled_ids: list[tuple[int, int | None]]


def _map_source_op_to_compiled_ops(
    source_level: str,
    source_op_id: int,
    debug_info_records: list[DebugInfoRecord],
) -> dict[int, tuple[int, int, int | None]]:
    """
    Map a source operation to its compiled ODIX operations.

    Maps each output of a source-level operation (e.g., a PyTorch operation)
    to its corresponding compiled ODIX operation ID and output index.
    When multiple mappings exist for the same source output, keeps the one
    with the highest target_op_id (representing the final compiled form).

    Args:
        source_level: Source dialect level (e.g., "torch" for PyTorch, "coreai" for Core AI)
        source_op_id: Unique identifier of the source operation
        debug_info_records: Parsed debug information containing operation mappings

    Returns:
        Dictionary mapping source output index to (odix_id, odix_output_index, delegate_id).
        The delegate_id is currently always None (reserved for future delegate support).

    """
    compiled_ids: dict[int, tuple[int, int, int | None]] = {}

    logger.debug("Mapping %s.%d to ODIX", source_level, source_op_id)

    for record in debug_info_records:
        if not record.identifier.startswith("odix"):
            continue

        for op in record.operations:
            mappings = op.get_output_mappings(source_level)
            for mapping in mappings:
                # Filter mappings for the specific source operation
                if (
                    mapping.source_op_id == source_op_id
                    and mapping.target_level == "odix"
                ):
                    # Check if we already have a mapping for this source output
                    existing = compiled_ids.get(mapping.source_output)

                    # Only update if this is a new mapping or has a higher target_op_id
                    if existing is None or mapping.target_op_id > existing[0]:
                        compiled_ids[mapping.source_output] = (
                            mapping.target_op_id,
                            mapping.target_output,
                            None,
                        )
                        if existing is not None:
                            logger.debug(
                                "  %s.%d[%d] -> odix.%d[%d] (replaced odix.%d)",
                                source_level,
                                source_op_id,
                                mapping.source_output,
                                mapping.target_op_id,
                                mapping.target_output,
                                existing[0],
                            )
                        else:
                            logger.debug(
                                "  %s.%d[%d] -> odix.%d[%d]",
                                source_level,
                                source_op_id,
                                mapping.source_output,
                                mapping.target_op_id,
                                mapping.target_output,
                            )

    if not compiled_ids:
        logger.warning("No ODIX mapping found for %s.%d", source_level, source_op_id)

    return compiled_ids


def _create_operation_mappings(
    op_ids: Sequence[int],
    source_level: str,
    debug_info_records: list[DebugInfoRecord],
) -> _CompiledIdMappings:
    """
    Create bidirectional mappings between source and compiled operations.

    Builds mappings that allow translating between source-level operations (e.g., PyTorch)
    and their compiled ODIX representations. This enables capturing intermediate values
    from compiled models and mapping them back to source operations.

    Args:
        op_ids: List of source operation IDs to create mappings for
        source_level: Source dialect level (e.g., "torch" for PyTorch, "coreai" for Core AI)
        debug_info_records: Parsed debug information containing operation mappings

    Returns:
        _CompiledIdMappings containing:
        - target_to_source_output_map: Maps ODIX outputs back to source outputs via _MappingKey
        - all_compiled_ids: Unique list of (odix_id, delegate_id) pairs for all operations

    """
    target_to_source_output_map: dict[_MappingKey, tuple[int, int]] = {}
    all_compiled_ids: list[tuple[int, int | None]] = []

    for source_op_id in op_ids:
        compiled_ids = _map_source_op_to_compiled_ops(
            source_level,
            source_op_id,
            debug_info_records,
        )

        for source_output_idx, (
            odix_id,
            target_output,
            delegate_id,
        ) in compiled_ids.items():
            all_compiled_ids.append((odix_id, delegate_id))
            mapping_key = _MappingKey(odix_id, delegate_id, target_output)
            target_to_source_output_map[mapping_key] = (source_op_id, source_output_idx)

    return _CompiledIdMappings(target_to_source_output_map, all_compiled_ids)


def _convert_to_dict(
    op_ids: Sequence[int],
    results: Mapping[int, dict[int, NDArray[Any]]],
) -> dict[int, list[NDArray[Any] | None] | None]:
    """
    Convert output dictionary structure to list format.

    Transforms the internal sparse dictionary representation (keyed by output index)
    into a dense list format expected by the public API. Missing indices are filled
    with None values.

    Args:
        op_ids: List of operation IDs to process
        results: Mapping of operation_id -> {output_index -> array}

    Returns:
        Dictionary mapping operation_id -> list of arrays (or None if operation not found).
        Each list contains arrays ordered by output index, with None for missing outputs.

    """
    final_results: dict[int, list[NDArray[Any] | None] | None] = {}
    for op_id in op_ids:
        output_dict = results.get(op_id)
        if output_dict:
            max_idx = max(output_dict.keys())
            output_list: list[NDArray[Any] | None] = [
                output_dict.get(i, None) for i in range(max_idx + 1)
            ]
            final_results[op_id] = output_list
        else:
            final_results[op_id] = None

    return final_results


class IntermediateKind(Enum):
    """
    Type of intermediate value captured during model execution.

    Attributes:
        INPUT: Represents an operation input value
        OUTPUT: Represents an operation output value
        UNKNOWN: Fallback for unrecognized intermediate types

    """

    INPUT = "input"
    OUTPUT = "output"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "IntermediateKind":
        """Return UNKNOWN for unrecognized values to avoid lookup errors."""
        return cls.UNKNOWN


class Inspector(ABC):
    """
    Abstract base class for capturing intermediate operation values during model execution.

    Inspectors provide a unified interface for executing models and capturing outputs
    from specific operations. This is essential for debugging workflows such as
    bisection search, where intermediate values are compared to identify numerical issues.

    Different implementations handle various model formats (PyTorch, compiled Core AI, etc.).
    """

    # Type alias for operation identifiers (string for PyTorch FX nodes, int for Core AI debug IDs)
    OpID = Union[str, int]

    @abstractmethod
    async def get_intermediates_for_ops(
        self,
        op_ids: list[OpID],
        inputs: Any,
    ) -> dict[OpID, list[NDArray[Any] | None] | None]:
        """
        Execute model and capture intermediate outputs for specified operations.

        Args:
            op_ids: List of operation identifiers to capture outputs for
            inputs: Model inputs (format varies by implementation)

        Returns:
            Dictionary mapping each operation ID to a list of output arrays.
            None indicates the operation wasn't executed or produced no outputs.

        """

    @classmethod
    def convert_to_numpy(cls, arr: Any) -> NDArray[Any]:
        """
        Convert an array to numpy.ndarray format.

        This method allows different inspector implementations to handle their
        framework-specific array types (e.g., Core AI NDArray).

        Args:
            arr: Array to convert (framework-specific or numpy array)

        Returns:
            NumPy array

        """
        # Default implementation: assume it's already numpy or numpy-compatible
        return np.asarray(arr)


class CachingInspector(Inspector):
    """
    Inspector decorator that caches intermediate values to avoid redundant execution.

    Maintains a cache of previously captured intermediate values, significantly
    improving performance when querying the same operations multiple times (common
    in bisection search). The cache is automatically invalidated when inputs change.

    This is a transparent wrapper - it preserves the same interface as the underlying
    inspector while adding caching behavior. When max_cache_size is set, uses LRU
    (Least Recently Used) eviction policy.
    """

    def __init__(self, inspector: Inspector, max_cache_size: int | None = None):
        """
        Initialize the caching inspector.

        Args:
            inspector: Underlying inspector instance to wrap with caching
            max_cache_size: Maximum number of entries to keep in cache. If None, cache
                          size is unlimited. When limit is reached, least recently used
                          entries are evicted (LRU policy).

        """
        self._inspector = inspector
        self._cache: OrderedDict[Inspector.OpID, list[NDArray[Any] | None] | None] = (
            OrderedDict()
        )
        self._current_inputs: Any = None
        self._max_cache_size = max_cache_size

    @staticmethod
    def _inputs_equal(inputs1: Any, inputs2: Any) -> bool:
        """
        Compare inputs for cache invalidation, handling containers properly.

        For tuple/list/dict structures, compares contents rather than object identity.
        For tensors and other objects, falls back to object identity comparison.

        Args:
            inputs1: First input to compare
            inputs2: Second input to compare

        Returns:
            True if inputs should be considered equal for caching purposes
        """
        if inputs1 is inputs2:
            return True

        if type(inputs1) is not type(inputs2):
            return False

        if isinstance(inputs1, dict):
            if inputs1.keys() != inputs2.keys():
                return False
            return all(
                CachingInspector._inputs_equal(inputs1[k], inputs2[k])
                for k in inputs1.keys()
            )
        elif isinstance(inputs1, (tuple, list)):
            if len(inputs1) != len(inputs2):
                return False
            return all(
                CachingInspector._inputs_equal(a, b) for a, b in zip(inputs1, inputs2)
            )
        else:
            # For tensors and other objects, use object identity
            return inputs1 is inputs2

    @staticmethod
    def _copy_inputs(inputs: Any) -> Any:
        """
        Create a copy of inputs for cache tracking.

        For tuple/list/dict structures, creates shallow copies.
        For other objects (like tensors), stores reference since deep copying
        tensors can be expensive and we only need to track identity changes.

        Args:
            inputs: Input to copy

        Returns:
            Copy of the input suitable for cache tracking
        """
        if isinstance(inputs, dict):
            return {k: CachingInspector._copy_inputs(v) for k, v in inputs.items()}
        elif isinstance(inputs, tuple):
            return tuple(CachingInspector._copy_inputs(item) for item in inputs)
        elif isinstance(inputs, list):
            return [CachingInspector._copy_inputs(item) for item in inputs]
        else:
            # For tensors and other objects, just return the reference
            # We rely on identity comparison for these
            return inputs

    async def get_intermediates_for_ops(
        self,
        op_ids: list[Inspector.OpID],
        inputs: Any,
    ) -> dict[Inspector.OpID, list[NDArray[Any] | None] | None]:
        """
        Retrieve intermediate outputs with automatic caching.

        Returns cached values when available, only executing the model for operations
        not yet in the cache. Automatically clears the cache when inputs change.
        Implements LRU eviction when max_cache_size is set.

        Args:
            op_ids: List of operation identifiers to capture outputs for
            inputs: Model inputs (cache is invalidated if these change)

        Returns:
            Dictionary mapping operation IDs to output arrays (cached or freshly captured)

        """
        if not self._inputs_equal(self._current_inputs, inputs):
            self._cache.clear()
            self._current_inputs = self._copy_inputs(inputs)

        uncached_ops = [op_id for op_id in op_ids if op_id not in self._cache]

        if uncached_ops:
            results = await self._inspector.get_intermediates_for_ops(
                uncached_ops,
                inputs,
            )
            if results is not None:
                for op_id, value in results.items():
                    if (
                        self._max_cache_size is not None
                        and len(self._cache) >= self._max_cache_size
                        and op_id not in self._cache
                    ):
                        self._cache.popitem(last=False)
                    self._cache[op_id] = value

        result = {}
        for op_id in op_ids:
            if op_id in self._cache:
                self._cache.move_to_end(op_id)
                result[op_id] = self._cache[op_id]
            else:
                result[op_id] = None
        return result

    def clear_cache(self) -> None:
        """Clear all cached intermediate values and reset input tracking."""
        self._cache.clear()
        self._current_inputs = None


class TorchFXInspector(Inspector):
    """
    Inspector for PyTorch ExportedProgram models.

    Executes a PyTorch ExportedProgram and captures intermediate values at specified
    FX graph nodes using a custom interpreter. This is used for debugging PyTorch
    models before compilation.

    The inspector works at the FX graph level, where operation IDs are node names.
    """

    def __init__(self, exported_program: torch.export.ExportedProgram):
        """
        Initialize the PyTorch FX inspector.

        Args:
            exported_program: PyTorch ExportedProgram to execute and inspect

        """
        self.exported_program = exported_program

    async def get_intermediates_for_ops(
        self,
        op_ids: list[Inspector.OpID],
        inputs: Any,
    ) -> dict[Inspector.OpID, list[NDArray[Any] | None] | None]:
        """
        Capture intermediate values at specified FX graph nodes.

        Executes the PyTorch model using a custom FX interpreter that captures
        intermediate values via callbacks. All outputs are converted to NumPy arrays.

        Args:
            op_ids: List of FX node names (operation IDs) to capture
            inputs: Tuple of input tensors matching the model's expected signature

        Returns:
            Dictionary mapping node names to lists of output arrays (as NumPy).
            None indicates a node wasn't executed or produced no outputs.

        """
        requested_nodes = set(op_ids)
        results: dict[Inspector.OpID, list[NDArray[Any] | None] | None] = {}

        def capture_callback(node: torch.fx.Node, result: Any) -> None:
            """Invoke callback for each node during interpretation."""
            if node.name in requested_nodes:
                if isinstance(result, (tuple, list)):
                    results[node.name] = [
                        self.__class__.convert_to_numpy(r) for r in result
                    ]
                else:
                    results[node.name] = [self.__class__.convert_to_numpy(result)]

        interpreter = _TorchFXNodeValueInterpreter(
            self.exported_program.module(),
            callback=capture_callback,
        )

        interpreter.run(inputs)

        for node_name in op_ids:
            if node_name not in results:
                results[node_name] = None

        return results

    @classmethod
    def convert_to_numpy(cls, value: Any) -> NDArray[Any]:
        """
        Convert PyTorch tensors or other values to NumPy arrays.

        Args:
            value: Value to convert (torch.Tensor, np.ndarray, or scalar)

        Returns:
            NumPy array (detached from PyTorch computation graph if applicable)

        """
        # Import torch here to avoid requiring it when not using PyTorch models
        import torch  # noqa: PLC0415

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)


class CoreAIInspector(Inspector):
    """
    Inspector for Core AI Runtime deployed models.

    Captures intermediate operation outputs from models deployed via Core AI Runtime.
    Uses the IntermediateLogger to intercept values during execution and map them
    back to source-level operations using debug info.

    Operation IDs are integer debug IDs from the source dialect (e.g., PyTorch operation IDs).
    """

    IntermediateCallback = Callable[
        [
            list[coreai.runtime._NDArray | None],
            str,
            coreai.runtime.CompileIdentifiers,
        ],
        None,
    ]

    def __init__(
        self,
        model: coreai.runtime.AIModel,
        function_name: str = "main",
        temp_dir: Any = None,
    ):
        """
        Initialize the Core AI Runtime inspector.

        Args:
            model: Loaded AIModel instance to execute and inspect
            function_name: Inference function name to execute (default: "main")
            temp_dir: Optional TemporaryDirectory to keep alive for the model's lifetime

        """
        self._model = model
        self._function_name = function_name
        self._last_outputs: dict[str, NDArray[Any]] | None = None
        self._temp_dir = temp_dir  # Keep temp directory alive

        # Load debug info from model
        debug_infos_bytes = model._debug_infos
        self._debug_info_records = parse_debug_infos(debug_infos_bytes)

        self._source_level = "coreai"

    def _build_mapping_and_compile_ids(
        self,
        op_ids: Sequence[int],
    ) -> tuple[
        dict[_MappingKey, tuple[int, int]],
        list[coreai.runtime.CompileIdentifiers],
    ]:
        """
        Build output mappings and compile identifiers for requested source operations.

        Creates the mapping infrastructure needed to translate captured compiled operation
        outputs back to their source operation equivalents.

        Args:
            op_ids: List of source operation IDs to create mappings for

        Returns:
            Tuple of (output_mapping, compile_identifiers) where:
            - output_mapping: Maps compiled outputs back to source outputs
            - compile_identifiers: List of unique compiled operation IDs to capture

        """
        mappings = _create_operation_mappings(
            op_ids,
            self._source_level,
            self._debug_info_records,
        )

        # Get unique compiled IDs preserving insertion order (dict.fromkeys for stable deduplication)
        unique_compiled_ids = dict.fromkeys(mappings.all_compiled_ids)
        all_compile_ids = [
            coreai.runtime.CompileIdentifiers(odix_id, delegate_id)
            for odix_id, delegate_id in unique_compiled_ids
        ]

        return mappings.target_to_source_output_map, all_compile_ids

    def _create_capture_callback(
        self,
        results: defaultdict[int, dict[int, NDArray[Any]]],
        odix_output_to_source_map: dict[_MappingKey, tuple[int, int]],
    ) -> "CoreAIInspector.IntermediateCallback":
        """
        Create callback function for Core AI Runtime IntermediateLogger.

        The callback is invoked during model execution for each intermediate value.
        It processes and stores the values in the results dictionary. Core AI provides
        all outputs for an operation at once (as a list).

        Args:
            results: Storage dictionary for captured values (source_op_id -> {output_index -> array})
            odix_output_to_source_map: Mapping from compiled to source operation outputs

        Returns:
            Callback function compatible with Core AI Runtime IntermediateLogger interface

        """

        def capture_callback(
            intermediates: list[coreai.runtime._NDArray | None],
            kind: str,
            compile_ids: coreai.runtime.CompileIdentifiers,
        ) -> None:
            kind_type = IntermediateKind(kind)

            if kind_type == IntermediateKind.UNKNOWN:
                msg = f"Unknown intermediate kind: {kind}"
                raise ValueError(msg)

            if kind_type != IntermediateKind.OUTPUT:
                return

            logger.debug(
                "Capturing %s for odix.%d (delegate=%s)",
                kind,
                compile_ids.id,
                compile_ids.delegate_id,
            )

            for odix_output_idx, intermediate in enumerate(intermediates):
                mapping_key = _MappingKey(
                    compile_ids.id,
                    compile_ids.delegate_id,
                    odix_output_idx,
                )
                if mapping_key in odix_output_to_source_map:
                    source_op_id, source_output_idx = odix_output_to_source_map[
                        mapping_key
                    ]

                    if source_output_idx in results[source_op_id]:
                        msg = f"Multiple compile_ids map to the same source operation output: source_op_id={source_op_id}, source_output_idx={source_output_idx}"
                        raise ValueError(msg)
                    if intermediate is not None:
                        # Convert _NDArray (internal Core AI runtime type) to NDArray wrapper then to numpy
                        ndarray = coreai.runtime.NDArray(intermediate)
                        results[source_op_id][source_output_idx] = (
                            self.__class__.convert_to_numpy(ndarray)
                        )
                        logger.debug(
                            "  odix.%d[%d] -> source.%d[%d] shape=%s",
                            compile_ids.id,
                            odix_output_idx,
                            source_op_id,
                            source_output_idx,
                            ndarray.numpy().shape,
                        )
                    else:
                        logger.warning(
                            "  Intermediate is None for odix.%d[%d] -> source.%d[%d]",
                            compile_ids.id,
                            odix_output_idx,
                            source_op_id,
                            source_output_idx,
                        )

        return capture_callback

    async def get_intermediates_for_ops(
        self,
        op_ids: list[Inspector.OpID],
        inputs: Any,
    ) -> dict[Inspector.OpID, list[NDArray[Any] | None] | None]:
        """
        Capture intermediate outputs from Core AI Runtime model execution.

        Executes the model with an IntermediateLogger that captures values at
        requested operations and maps them back to source-level operations.

        Args:
            op_ids: List of source operation debug IDs to capture
            inputs: Model inputs (dictionary mapping input names to NDArray or numpy arrays)

        Returns:
            Dictionary mapping operation IDs to lists of output arrays.
            None indicates an operation wasn't executed or produced no outputs.

        Raises:
            TypeError: If inputs is not a dictionary

        """
        if not isinstance(inputs, dict):
            msg = "inputs must be a dictionary mapping input names to NDArray"
            raise TypeError(msg)

        # Convert inputs to NDArray objects if needed
        ndarray_inputs = {}
        for name, value in inputs.items():
            if not isinstance(value, coreai.runtime.NDArray):
                ndarray_inputs[name] = coreai.runtime.NDArray(value)
            else:
                ndarray_inputs[name] = value

        int_op_ids = cast(list[int], op_ids)

        logger.debug("Requested source op_ids: %s", int_op_ids)

        odix_output_to_source_map, all_compile_ids = (
            self._build_mapping_and_compile_ids(int_op_ids)
        )

        logger.debug("Found %d compile_ids to capture", len(all_compile_ids))
        for compile_id in all_compile_ids:
            logger.debug(
                "  - odix_id=%s, delegate_id=%s",
                compile_id.id,
                compile_id.delegate_id,
            )

        # If no compile IDs found, return None for all requested ops
        if not all_compile_ids:
            logger.warning("No compile_ids found for requested operations")
            return dict.fromkeys(op_ids)

        results: defaultdict[int, dict[int, NDArray[Any]]] = defaultdict(dict)

        capture_callback = self._create_capture_callback(
            results,
            odix_output_to_source_map,
        )

        intermediate_logger = coreai.runtime.IntermediateLogger(
            requested_intermediates=all_compile_ids,
            callback=capture_callback,
            is_enabled=True,
        )

        inference_function = self._model.load_function(
            self._function_name,
            intermediate_logger=intermediate_logger,
        )

        outputs = await inference_function(inputs=ndarray_inputs)

        self._last_outputs = {name: array.numpy() for name, array in outputs.items()}

        logger.debug("Successfully captured %d source operations", len(results))

        # int is a subtype of Inspector.OpID (int | str), so this is safe
        return _convert_to_dict(int_op_ids, results)  # type: ignore[return-value]

    @classmethod
    def convert_to_numpy(cls, arr: Any) -> NDArray[Any]:
        """
        Convert Core AI NDArray to numpy array.

        Args:
            arr: Core AI NDArray object

        Returns:
            NumPy array

        """
        return np.asarray(arr.numpy())

    @property
    def last_outputs(self) -> dict[str, NDArray[Any]] | None:
        """
        Get final model outputs from the most recent execution.

        Returns:
            Dictionary mapping output names to NumPy arrays, or None if model
            hasn't been executed yet

        """
        return self._last_outputs

    @staticmethod
    def get_compile_identifiers_for_op(
        source_level: str,
        source_op_id: int,
        debug_info_records: list[DebugInfoRecord],
    ) -> dict[int, coreai.runtime.CompileIdentifiers]:
        """
        Get compiled operation identifiers for a source operation.

        Maps each output of a source operation to its CompileIdentifiers
        (used by the Core AI Runtime).

        Args:
            source_level: Source dialect level ("torch" for PyTorch, "coreai" for Core AI)
            source_op_id: Source operation ID to look up
            debug_info_records: Debug information containing operation mappings

        Returns:
            Dictionary mapping output index to CompileIdentifiers

        """
        compiled_ids = _map_source_op_to_compiled_ops(
            source_level,
            source_op_id,
            debug_info_records,
        )
        return {
            output_idx: coreai.runtime.CompileIdentifiers(
                odix_id,
                delegate_id,
            )
            for output_idx, (
                odix_id,
                _target_output,
                delegate_id,
            ) in compiled_ids.items()
        }
