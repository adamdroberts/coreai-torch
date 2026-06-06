# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared configuration for compression tests."""

from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark compression tests as 'compression' with flaky reruns."""
    for item in items:
        if Path(item.fspath).is_relative_to(_THIS_DIR):
            item.add_marker(pytest.mark.compression)
            item.add_marker(pytest.mark.flaky(reruns=3))
