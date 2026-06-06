# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared configuration for operator tests."""

from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).parent
_CUSTOM_OPS_FILE = _THIS_DIR / "test_custom_ops.py"
_IR_TEST_FILE = _THIS_DIR / "test_ops_ir.py"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark operator tests as 'ops'. Mark custom_ops (compression) as flaky."""
    for item in items:
        if Path(item.fspath).is_relative_to(_THIS_DIR):
            item.add_marker(pytest.mark.ops)
            if Path(item.fspath) == _CUSTOM_OPS_FILE:
                item.add_marker(pytest.mark.flaky(reruns=3))
            if Path(item.fspath) == _IR_TEST_FILE:
                item.add_marker(pytest.mark.ir)
