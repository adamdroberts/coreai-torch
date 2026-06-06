# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# This utility file is only used by _compression.py and is not intended for external use.
# It will be migrated to other repo along with the _compression.py.
# Disable ruff check for now.
# ruff: noqa: D401, C901, PLW1641, N801, N804, D200, EM102, PLC0415

import builtins
import logging
import math
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import sympy as sm  # type: ignore[import-untyped]

# The scalar-type machinery below is adapted from coremltools (BSD-3-Clause).
logger = logging.getLogger(__name__)


class Type:
    __slots__ = ("name", "tparam", "python_class")

    def __init__(
        self,
        name: builtins.str,
        tparam: Optional[list] = None,
        python_class: Optional[type] = None,
    ) -> None:
        if tparam is None:
            tparam = []
        assert isinstance(name, builtins.str)
        assert isinstance(tparam, list)
        self.name = name
        self.tparam = tparam
        self.python_class = python_class

    def __hash__(self) -> builtins.int:
        return hash((self.name, tuple(self.tparam)))

    def __eq__(self, other: object) -> builtins.bool:
        if not isinstance(other, Type):
            return NotImplemented
        return self.name == other.name and self.tparam == other.tparam

    def __ne__(self, other: object) -> builtins.bool:
        return not self.__eq__(other)

    def __repr__(self) -> builtins.str:
        ret = self.name
        if len(self.tparam) > 0:
            ret += "[" + ",".join(repr(x) for x in self.tparam) + "]"
        return ret

    def __str__(self) -> builtins.str:
        return self.__repr__()


def class_annotate() -> Any:
    """No-op decorator preserved for source compatibility with the int wrapper."""

    def decorator(cls: type) -> type:
        return cls

    return decorator


def get_type_info(t: object) -> Type:
    if hasattr(t, "__type_info__"):
        ret = t.__type_info__()  # type: ignore[attr-defined]
        assert ret.python_class is not None
        return ret
    if isinstance(t, type):
        return Type(t.__name__, python_class=t)
    raise TypeError(f"Unsupported type {t}")


class types_bool:
    def __init__(self, v: object = False) -> None:
        self.val = v

    @classmethod
    def __type_info__(cls) -> Type:
        return Type("bool", python_class=cls)

    def __eq__(self, other: object) -> "types_bool":  # type: ignore[override]
        return types_bool(self.val == other.val)  # type: ignore[attr-defined]

    def __ne__(self, other: object) -> "types_bool":  # type: ignore[override]
        return types_bool(self.val != other.val)  # type: ignore[attr-defined]

    def __bool__(self) -> builtins.bool:
        return builtins.bool(self.val)


class types_str:
    def __init__(self, v: object = "") -> None:
        self.val = v

    @classmethod
    def __type_info__(cls) -> Type:
        return Type("str", python_class=cls)


def _make_float(name: builtins.str, width: builtins.int) -> type:
    class _Float:
        _width = width

        def __init__(self, v: object = 0.0) -> None:
            self.val = v

        @classmethod
        def __type_info__(cls) -> Type:
            return Type(name, python_class=cls)

        @classmethod
        def get_bitwidth(cls) -> builtins.int:
            return cls._width

    _Float.__name__ = name
    _Float.__qualname__ = name
    return _Float


types_fp16 = _make_float("fp16", 16)
types_fp32 = _make_float("fp32", 32)
types_fp64 = _make_float("fp64", 64)


def _make_complex(name: builtins.str, width: builtins.int) -> type:
    class _Complex:
        _width = width

        def __init__(self, real: object = 0.0, imag: object = 0.0) -> None:
            self.val = builtins.complex(real, imag)  # type: ignore[arg-type]

        @classmethod
        def __type_info__(cls) -> Type:
            return Type(name, python_class=cls)

        @classmethod
        def get_bitwidth(cls) -> builtins.int:
            return cls._width

    _Complex.__name__ = name
    _Complex.__qualname__ = name
    return _Complex


types_complex64 = _make_complex("complex64", 64)
types_complex128 = _make_complex("complex128", 128)


# Map of numpy scalar/dtype -> compression int type. Populated by
# ``register_numpy_int_mapping`` (called below once the int wrappers exist).
_NUMPY_INT_TO_BUILTIN: dict[Any, type] = {}


def register_numpy_int_mapping(mapping: dict[Any, type]) -> None:
    """Register numpy integer dtype → builtin int type mapping."""
    _NUMPY_INT_TO_BUILTIN.update(mapping)


def numpy_type_to_builtin_type(nptype: Any) -> type:  # noqa: C901, PLR0911, PLR0912
    """Convert a numpy dtype or scalar type to its builtin compression type."""
    if isinstance(nptype, np.dtype):
        metadata = nptype.metadata
        if metadata is not None and SUB_BYTE_DTYPE_METADATA_KEY in metadata:
            return metadata[SUB_BYTE_DTYPE_METADATA_KEY]  # type: ignore[no-any-return]
        if nptype in _NUMPY_INT_TO_BUILTIN:
            return _NUMPY_INT_TO_BUILTIN[nptype]
        nptype = nptype.type

    if issubclass(nptype, (builtins.bool, np.bool_)):
        return types_bool
    if issubclass(nptype, np.uint8):
        return _NUMPY_INT_TO_BUILTIN[np.uint8]
    if issubclass(nptype, np.int8):
        return _NUMPY_INT_TO_BUILTIN[np.int8]
    if issubclass(nptype, np.uint16):
        return _NUMPY_INT_TO_BUILTIN[np.uint16]
    if issubclass(nptype, np.int16):
        return _NUMPY_INT_TO_BUILTIN[np.int16]
    if issubclass(nptype, np.uint32):
        return _NUMPY_INT_TO_BUILTIN[np.uint32]
    if issubclass(nptype, np.int32):
        return _NUMPY_INT_TO_BUILTIN[np.int32]
    if issubclass(nptype, np.uint64):
        return _NUMPY_INT_TO_BUILTIN[np.uint64]
    if issubclass(nptype, np.int64):
        return _NUMPY_INT_TO_BUILTIN[np.int64]
    if issubclass(nptype, builtins.int) or nptype == builtins.int:
        return _NUMPY_INT_TO_BUILTIN[np.int32]
    if issubclass(nptype, np.object_):
        return _NUMPY_INT_TO_BUILTIN[np.int32]
    if issubclass(nptype, np.float16):
        return types_fp16
    if issubclass(nptype, (np.float32, np.single)) or nptype == builtins.float:
        return types_fp32
    if issubclass(nptype, (np.float64, np.double)):
        return types_fp64
    if issubclass(nptype, np.complex64):
        return types_complex64
    if issubclass(nptype, (np.complex128, builtins.complex)):
        return types_complex128
    if issubclass(nptype, (builtins.str, np.bytes_, np.str_)):
        return types_str
    raise TypeError(f"Unsupported numpy type: {nptype}.")


@runtime_checkable
class Dtype(Protocol):
    """Structural contract used by the ``*_by_dtype`` helpers in ``utils.py``.

    Satisfied by the int wrapper classes produced by ``make_int`` (which
    expose both ``get_bitwidth`` and ``is_unsigned``) and by the float
    wrappers ``types_fp16/32/64`` (which expose only ``get_bitwidth``).
    Callers that need ``is_unsigned`` guard with ``is_int(dtype)`` first;
    invoking it on a non-int dtype raises ``AttributeError`` at runtime.
    """

    @classmethod
    def get_bitwidth(cls) -> builtins.int: ...

    @classmethod
    def is_unsigned(cls) -> builtins.bool: ...


def make_int(width_: int, unsigned_: str) -> Any:
    @class_annotate()
    class int:
        _width = width_
        _unsigned = unsigned_

        @classmethod
        def width(self) -> builtins.int:
            return self._width

        @classmethod
        def unsigned(self) -> str:
            return self._unsigned

        def __init__(self, v: Any = 0) -> None:
            self._val = v

        @property
        def val(self) -> Any:
            return self._val

        @val.setter
        def val(self, v: object) -> None:
            if not isinstance(v, (np.generic, np.ndarray, sm.Basic)):
                try:
                    v = np.array(v)
                except Exception:
                    raise ValueError(
                        f"types should have value of numpy type or Symbols, got {type(v)} instead",
                    )

            if isinstance(v, sm.Basic):
                self._val = v
            elif isinstance(v, np.integer):
                v_type = numpy_type_to_builtin_type(v.dtype)
                if v_type.get_bitwidth() <= self.get_bitwidth() and (
                    v >= 0 or (v < 0 and not self.is_unsigned())
                ):
                    self._val = v
                else:
                    self._val = v.astype(nptype_from_builtin(self.__class__))
                    logger.warning(
                        f"Saving value type of {v.dtype} into a builtin type of "
                        f"{builtin_to_string(self.__class__)}, might overflow or loses precision!",
                    )
            else:
                self._val = v.astype(nptype_from_builtin(self.__class__))
                logger.warning(
                    f"Saving value type of {v.dtype} into a builtin type of "
                    f"{builtin_to_string(self.__class__)}, might be incompatible or loses precision!",
                )

        @classmethod
        def __type_info__(cls) -> Type:
            return Type(cls._unsigned + "int" + str(cls._width), python_class=cls)

        @classmethod
        def get_bitwidth(cls) -> builtins.int:
            return cls._width

        @classmethod
        def is_unsigned(cls) -> bool:
            return cls._unsigned == "u"

        def __add__(self, other: "int") -> Any:
            assert isinstance(other, int)
            return int(self.val + other.val)

        def __sub__(self, other: "int") -> Any:
            assert isinstance(other, int)
            return int(self.val - other.val)

        def __mul__(self, other: "int") -> Any:
            assert isinstance(other, int)
            return int(self.val * other.val)

        def __div__(self, other: "int") -> Any:
            assert isinstance(other, int)
            return int(self.val // other.val)

        def __mod__(self, other: "int") -> Any:
            assert isinstance(other, int)
            return int(self.val % other.val)

        def __lt__(self, other: "int") -> types_bool:
            return types_bool(self.val < other.val)

        def __gt__(self, other: "int") -> types_bool:
            return types_bool(self.val > other.val)

        def __le__(self, other: "int") -> types_bool:
            return types_bool(self.val <= other.val)

        def __ge__(self, other: "int") -> types_bool:
            return types_bool(self.val >= other.val)

        def __eq__(self, other: "int") -> types_bool:  # type: ignore[override]
            return types_bool(self.val == other.val)

        def __ne__(self, other: "int") -> types_bool:  # type: ignore[override]
            return types_bool(self.val != other.val)

        def __bool__(self) -> types_bool:
            return self.val != 0

        def __int__(self) -> Any:
            return int(self)

        def __double__(self) -> types_fp32:
            return float(self.val)

        def __str__(self) -> types_str:
            return str(self.val)

        def __log__(self) -> types_fp32:
            return math.log(self.val)

        def __exp__(self) -> types_fp32:
            return math.exp(self.val)

        def __neg__(self) -> Any:
            return int(-self.val)

    return int


int2 = make_int(2, "")
int4 = make_int(4, "")
int8 = make_int(8, "")
int16 = make_int(16, "")
int32 = make_int(32, "")
int64 = make_int(64, "")

uint1 = make_int(1, "u")
uint2 = make_int(2, "u")
uint3 = make_int(3, "u")
uint4 = make_int(4, "u")
uint6 = make_int(6, "u")
uint8 = make_int(8, "u")
uint16 = make_int(16, "u")
uint32 = make_int(32, "u")
uint64 = make_int(64, "u")
uint = uint64

_INT_TYPES = (
    int2,
    int4,
    int8,
    int16,
    int32,
    int64,
    uint1,
    uint2,
    uint3,
    uint4,
    uint6,
    uint8,
    uint16,
    uint32,
    uint64,
)

# The key name for storing type info in `np.dtype.metadata`.
SUB_BYTE_DTYPE_METADATA_KEY = "true_dtype"
# Uses np.int8/uint8 as np doesn't natively support sub-byte type (such as int4/uint4) yet.
np_int2_dtype = np.dtype(np.int8, metadata={SUB_BYTE_DTYPE_METADATA_KEY: int2})
np_int4_dtype = np.dtype(np.int8, metadata={SUB_BYTE_DTYPE_METADATA_KEY: int4})
np_uint1_dtype = np.dtype(np.uint8, metadata={SUB_BYTE_DTYPE_METADATA_KEY: uint1})
np_uint2_dtype = np.dtype(np.uint8, metadata={SUB_BYTE_DTYPE_METADATA_KEY: uint2})
np_uint3_dtype = np.dtype(np.uint8, metadata={SUB_BYTE_DTYPE_METADATA_KEY: uint3})
np_uint4_dtype = np.dtype(np.uint8, metadata={SUB_BYTE_DTYPE_METADATA_KEY: uint4})
np_uint6_dtype = np.dtype(np.uint8, metadata={SUB_BYTE_DTYPE_METADATA_KEY: uint6})

_SUB_BYTE_TYPES = (int2, int4, uint1, uint2, uint3, uint4, uint6)

_TYPES_TO_STRINGS = {
    types_bool: "bool",
    int2: "int2",
    int4: "int4",
    int8: "int8",
    int16: "int16",
    int32: "int32",
    int64: "int64",
    uint1: "uint1",
    uint2: "uint2",
    uint3: "uint3",
    uint4: "uint4",
    uint6: "uint6",
    uint8: "uint8",
    uint16: "uint16",
    uint32: "uint32",
    uint64: "uint64",
    types_fp16: "fp16",
    types_fp32: "fp32",
    types_fp64: "fp64",
    types_complex64: "complex64",
    types_complex128: "complex128",
    types_str: "string",
}

_TYPES_TO_NPTYPES = {
    types_bool: np.bool_,
    int2: np_int2_dtype,
    int4: np_int4_dtype,
    int8: np.int8,
    int16: np.int16,
    int32: np.int32,
    int64: np.int64,
    uint1: np_uint1_dtype,
    uint2: np_uint2_dtype,
    uint3: np_uint3_dtype,
    uint4: np_uint4_dtype,
    uint6: np_uint6_dtype,
    uint8: np.uint8,
    uint16: np.uint16,
    uint32: np.uint32,
    uint64: np.uint64,
    types_fp16: np.float16,
    types_fp32: np.float32,
    types_fp64: np.float64,
    types_complex64: np.complex64,
    types_complex128: np.complex128,
    types_str: np.str_,
}

_STRINGS_TO_TYPES = {v: k for k, v in _TYPES_TO_STRINGS.items()}

# Wire numpy scalar dtypes to the int wrappers above so the local
# ``numpy_type_to_builtin_type`` (defined above) can resolve integer inputs.
register_numpy_int_mapping(
    {
        np.dtype(np.int8): int8,
        np.dtype(np.int16): int16,
        np.dtype(np.int32): int32,
        np.dtype(np.int64): int64,
        np.dtype(np.uint8): uint8,
        np.dtype(np.uint16): uint16,
        np.dtype(np.uint32): uint32,
        np.dtype(np.uint64): uint64,
        np.int8: int8,
        np.int16: int16,
        np.int32: int32,
        np.int64: int64,
        np.uint8: uint8,
        np.uint16: uint16,
        np.uint32: uint32,
        np.uint64: uint64,
    },
)


def string_to_builtin(s: str) -> Any:
    """
    Given a str, return its corresponding builtin type.
    """
    return _STRINGS_TO_TYPES[s]


def builtin_to_string(builtin_type: type) -> str:
    """
    Given a builtin type, return its corresponding string representation.
    """
    if is_dict(builtin_type):
        return "dict"
    return _TYPES_TO_STRINGS[builtin_type]


def nptype_from_builtin(btype: type) -> np.dtype[Any]:
    """
    Given a builtin type, return its corresponding Numpy dtype.
    """
    return _TYPES_TO_NPTYPES[btype]  # type: ignore[return-value]


def is_dict(t: object) -> bool:
    if t is None:
        return False
    try:
        type_info = get_type_info(t).name
    except TypeError:
        return False
    return type_info == "dict"  # type: ignore[no-any-return]


def is_int(t: object) -> bool:
    return any(t is i or isinstance(t, i) for i in _INT_TYPES)


def is_signed_int(t: object) -> bool:
    return bool(t._unsigned == "")  # type: ignore[attr-defined]


def is_unsigned_int(t: object) -> bool:
    return bool(t._unsigned == "u")  # type: ignore[attr-defined]


def is_sub_byte(t: object) -> bool:
    """Determines if a type (or instance) is sub-byte (less than 8-bit data type)."""
    return t in _SUB_BYTE_TYPES or isinstance(t, _SUB_BYTE_TYPES)


class LUTDtype(str, Enum):
    """
    Enum for LUT (lookup table) data types in palettization.

    Supports both enum and string values
    """

    INT8 = "int8"
    FLOAT8E4M3 = "fp8_e4m3"

    @classmethod
    def from_value(cls, value: object) -> Optional["LUTDtype"]:
        """
        Convert a string or enum value to LUTDtype enum.

        Args:
            value: Either a LUTDtype enum, a string ("int8"), or None

        Returns:
            LUTDtype enum or None

        Raises:
            ValueError: If value is not None and not a valid LUT dtype

        """
        if value is None:
            return None

        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                valid_values = [e.value for e in cls]
                raise ValueError(
                    f"Invalid LUT dtype: '{value}'. Must be one of {valid_values} or None.",
                )

        raise ValueError(
            f"Invalid LUT dtype type: {type(value)}. Must be a string, LUTDtype enum, or None.",
        )
