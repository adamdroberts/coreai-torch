#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""freqop — Compare the frequency of operations in two Core AI programs.

Usage:
    python tools/freqop/freqop.py <file_a> <file_b>
    python tools/freqop/freqop.py <file_a>              # just print op counts for one file

Loads AIModel asset (.aimodel) directories via coreai's AIProgram,
which automatically converts the serialized form to the coreai dialect.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from coreai.authoring import AIModelAsset

COREAI_DIALECT = "coreai."


def load_program(path: Path):
    """Load an AIModel asset (.aimodel) and return an AIProgram."""

    return AIModelAsset.load(path).program


def count_operations(
    program, *, collapse_composites: bool = False
) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    """Walk the operation tree and count coreai dialect ops.

    Returns (op_counts, composite_counts, composite_details) where:
    - op_counts: frequency of each coreai op in the program.
    - composite_counts: frequency of each composite type (e.g. gather_mm: 72).
    - composite_details: per-composite-type inner op breakdown (only populated
      when *collapse_composites* is True).

    When *collapse_composites* is True, composite graph bodies are NOT
    recursed into — each composite is treated as a single op in op_counts.
    The inner ops are instead collected into *composite_details*.
    When False (default), all ops inside composites are counted individually.
    """
    counts: Counter[str] = Counter()
    composites: Counter[str] = Counter()
    composite_details: dict[str, Counter[str]] = {}

    def _composite_name(op) -> str | None:
        """Extract the composite op name from a coreai.graph's composite_decl attribute."""
        if "composite_decl" not in op.attributes:
            return None
        # Format: #coreai.composite_declaration<"name" = {... }>
        attr_str = str(op.attributes["composite_decl"])
        start = attr_str.find('"')
        end = attr_str.find('"', start + 1)
        if start != -1 and end != -1:
            return attr_str[start + 1 : end]
        return None

    def walk(op, *, target: Counter[str] | None = None):
        dest = target if target is not None else counts
        name = op.name
        composite_body_target = None
        if name.startswith(COREAI_DIALECT):
            op_name = name[len(COREAI_DIALECT) :]
            if op_name == "graph":
                composite = _composite_name(op)
                if composite:
                    composites[composite] += 1
                    if collapse_composites:
                        op_name = f"composite.{composite}"
                        detail = composite_details.setdefault(composite, Counter())
                        composite_body_target = detail
                    else:
                        op_name = "graph"
                else:
                    op_name = "graph"
            dest[op_name] += 1
        if composite_body_target is not None:
            for region in op.regions:
                for block in region.blocks:
                    for nested_op in block.operations:
                        walk(nested_op, target=composite_body_target)
        else:
            for region in op.regions:
                for block in region.blocks:
                    for nested_op in block.operations:
                        walk(nested_op, target=target)

    walk(program.module.operation)
    return (
        dict(counts),
        dict(composites),
        {k: dict(v) for k, v in composite_details.items()},
    )


def plot_single(counts: dict[str, int], label: str) -> None:
    """Show a matplotlib bar chart for a single program's op counts."""
    import matplotlib.pyplot as plt

    ops = sorted(counts, key=counts.get, reverse=True)
    values = [counts[op] for op in ops]

    fig, ax = plt.subplots(figsize=(10, max(4, len(ops) * 0.35)))
    ax.barh(ops, values)
    ax.set_xlabel("Count")
    ax.set_title(f"Core AI op frequency — {label}")
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), va="center", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_comparison(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    label_a: str,
    label_b: str,
) -> None:
    """Show a grouped horizontal bar chart comparing two programs."""
    import matplotlib.pyplot as plt
    import numpy as np

    all_ops = sorted(
        set(counts_a) | set(counts_b),
        key=lambda op: max(counts_a.get(op, 0), counts_b.get(op, 0)),
        reverse=True,
    )
    values_a = [counts_a.get(op, 0) for op in all_ops]
    values_b = [counts_b.get(op, 0) for op in all_ops]

    y = np.arange(len(all_ops))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(4, len(all_ops) * 0.45)))
    ax.barh(y - bar_height / 2, values_a, bar_height, label=label_a)
    ax.barh(y + bar_height / 2, values_b, bar_height, label=label_b)
    ax.set_yticks(y)
    ax.set_yticklabels(all_ops)
    ax.set_xlabel("Count")
    ax.set_title("Core AI op frequency comparison")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_single(counts: dict[str, int], label: str) -> None:
    """Print op frequency table for a single program."""
    if not counts:
        print(f"No coreai operations found in {label}.")
        return

    max_op_len = max(len(op) for op in counts)
    header_op = "Operation".ljust(max_op_len)
    header_cnt = "Count"
    print(f"\n  {header_op}  {header_cnt}")
    print(f"  {'─' * max_op_len}  {'─' * len(header_cnt)}")
    for op, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op.ljust(max_op_len)}  {cnt}")
    print(f"\n  Total: {sum(counts.values())} ops  ({len(counts)} unique)")


def _build_comparison_data(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
) -> tuple[list[str], list[tuple[str, int, int, int]], int, int, int]:
    """Compute the sorted ops, per-row data, and totals for a comparison.

    Returns (all_ops, rows, total_a, total_b, diff_count) where each row is
    (op, count_a, count_b, delta).
    """
    all_ops = sorted(
        set(counts_a) | set(counts_b),
        key=lambda op: max(counts_a.get(op, 0), counts_b.get(op, 0)),
        reverse=True,
    )
    rows: list[tuple[str, int, int, int]] = []
    total_a = total_b = diff_count = 0
    for op in all_ops:
        a = counts_a.get(op, 0)
        b = counts_b.get(op, 0)
        total_a += a
        total_b += b
        delta = b - a
        if delta != 0:
            diff_count += 1
        rows.append((op, a, b, delta))
    return all_ops, rows, total_a, total_b, diff_count


def print_comparison(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    label_a: str,
    label_b: str,
) -> None:
    """Print a side-by-side comparison table with a delta column."""
    all_ops, rows, total_a, total_b, diff_count = _build_comparison_data(
        counts_a, counts_b
    )
    if not all_ops:
        print("No coreai operations found in either file.")
        return

    # ANSI background highlight codes
    RED_BG = "\033[48;2;50;10;10m\033[97m"  # muted dark red bg, bright white text
    GREEN_BG = "\033[48;2;10;40;10m\033[97m"  # muted dark green bg, bright white text
    RESET = "\033[0m"

    max_op_len = max(len(op) for op in all_ops)

    # Header
    print(
        f"\n  {'Operation'.ljust(max_op_len)}  {'Model 1':>8}  {'Model 2':>8}  {'Delta':>8}"
    )
    print(f"  {'─' * max_op_len}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    for op, a, b, delta in rows:
        delta_str = f"{delta:+d}" if delta != 0 else ""
        marker = " *" if delta != 0 else ""
        line = f"  {op.ljust(max_op_len)}  {a:>8}  {b:>8}  {delta_str:>8}{marker}"
        if delta < 0:
            line = f"{RED_BG}{line}{RESET}"
        elif delta > 0:
            line = f"{GREEN_BG}{line}{RESET}"
        print(line)

    # Summary
    total_delta = total_b - total_a
    print(f"  {'─' * max_op_len}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    delta_str = f"{total_delta:+d}" if total_delta != 0 else "0"
    print(f"  {'Total'.ljust(max_op_len)}  {total_a:>8}  {total_b:>8}  {delta_str:>8}")
    unique_a = len(counts_a)
    unique_b = len(counts_b)
    print(f"\n  Unique ops: {unique_a} vs {unique_b}")
    if diff_count == 0:
        print("  No differences found — programs have identical op frequencies.")
    else:
        print(f"  {diff_count} op(s) differ (marked with *).")
    print(f"\n  Model 1: {label_a}")
    print(f"  Model 2: {label_b}")


def format_comparison_txt(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    label_a: str,
    label_b: str,
) -> str:
    """Return a plain-text comparison table (no ANSI codes)."""
    all_ops, rows, total_a, total_b, diff_count = _build_comparison_data(
        counts_a, counts_b
    )
    if not all_ops:
        return "No coreai operations found in either file.\n"

    max_op_len = max(len(op) for op in all_ops)
    lines: list[str] = []

    lines.append(
        f"  {'Operation'.ljust(max_op_len)}  {'Model 1':>8}  {'Model 2':>8}  {'Delta':>8}"
    )
    lines.append(f"  {'─' * max_op_len}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    for op, a, b, delta in rows:
        delta_str = f"{delta:+d}" if delta != 0 else ""
        marker = " *" if delta != 0 else ""
        lines.append(
            f"  {op.ljust(max_op_len)}  {a:>8}  {b:>8}  {delta_str:>8}{marker}"
        )

    total_delta = total_b - total_a
    lines.append(f"  {'─' * max_op_len}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    delta_str = f"{total_delta:+d}" if total_delta != 0 else "0"
    lines.append(
        f"  {'Total'.ljust(max_op_len)}  {total_a:>8}  {total_b:>8}  {delta_str:>8}"
    )
    lines.append(f"\n  Unique ops: {len(counts_a)} vs {len(counts_b)}")
    if diff_count == 0:
        lines.append("  No differences found — programs have identical op frequencies.")
    else:
        lines.append(f"  {diff_count} op(s) differ (marked with *).")
    lines.append(f"\n  Model 1: {label_a}")
    lines.append(f"  Model 2: {label_b}")
    return "\n".join(lines) + "\n"


def format_single_txt(counts: dict[str, int], label: str) -> str:
    """Return a plain-text op frequency table (no ANSI codes)."""
    if not counts:
        return f"No coreai operations found in {label}.\n"

    max_op_len = max(len(op) for op in counts)
    lines: list[str] = []
    lines.append(f"  {'Operation'.ljust(max_op_len)}  {'Count'}")
    lines.append(f"  {'─' * max_op_len}  {'─' * 5}")
    for op, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {op.ljust(max_op_len)}  {cnt}")
    lines.append(f"\n  Total: {sum(counts.values())} ops  ({len(counts)} unique)")
    return "\n".join(lines) + "\n"


def format_comparison_html(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    label_a: str,
    label_b: str,
) -> str:
    """Return an HTML comparison table."""
    from html import escape

    all_ops, rows, total_a, total_b, diff_count = _build_comparison_data(
        counts_a, counts_b
    )
    if not all_ops:
        return "<p>No coreai operations found in either file.</p>\n"

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append("<title>freqop comparison</title>")
    parts.append("<style>")
    parts.append("body { font-family: monospace; margin: 2em; }")
    parts.append("table { border-collapse: collapse; }")
    parts.append(
        "th, td { padding: 4px 12px; text-align: right; border: 1px solid #ccc; }"
    )
    parts.append("th { background: #f0f0f0; }")
    parts.append("td:first-child, th:first-child { text-align: left; }")
    parts.append(".added { background: #d4edda; }")
    parts.append(".removed { background: #f8d7da; }")
    parts.append(".summary { margin-top: 1em; }")
    parts.append("</style></head><body>")
    parts.append("<h2>Core AI op frequency comparison</h2>")
    parts.append("<table><thead><tr>")
    parts.append("<th>Operation</th><th>Model 1</th><th>Model 2</th><th>Delta</th>")
    parts.append("</tr></thead><tbody>")

    for op, a, b, delta in rows:
        delta_str = f"{delta:+d}" if delta != 0 else ""
        css = ""
        if delta > 0:
            css = ' class="added"'
        elif delta < 0:
            css = ' class="removed"'
        parts.append(
            f"<tr{css}><td>{escape(op)}</td><td>{a}</td><td>{b}</td><td>{delta_str}</td></tr>"
        )

    total_delta = total_b - total_a
    delta_str = f"{total_delta:+d}" if total_delta != 0 else "0"
    parts.append(
        f"<tr style='font-weight:bold; border-top:2px solid #333'>"
        f"<td>Total</td><td>{total_a}</td><td>{total_b}</td><td>{delta_str}</td></tr>"
    )
    parts.append("</tbody></table>")

    parts.append('<div class="summary">')
    parts.append(f"<p>Unique ops: {len(counts_a)} vs {len(counts_b)}</p>")
    if diff_count == 0:
        parts.append(
            "<p>No differences found — programs have identical op frequencies.</p>"
        )
    else:
        parts.append(f"<p>{diff_count} op(s) differ.</p>")
    parts.append(f"<p>Model 1: {escape(label_a)}<br>Model 2: {escape(label_b)}</p>")
    parts.append("</div></body></html>")
    return "\n".join(parts) + "\n"


def format_single_html(counts: dict[str, int], label: str) -> str:
    """Return an HTML op frequency table."""
    from html import escape

    if not counts:
        return f"<p>No coreai operations found in {escape(label)}.</p>\n"

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append("<title>freqop</title>")
    parts.append("<style>")
    parts.append("body { font-family: monospace; margin: 2em; }")
    parts.append("table { border-collapse: collapse; }")
    parts.append(
        "th, td { padding: 4px 12px; text-align: right; border: 1px solid #ccc; }"
    )
    parts.append("th { background: #f0f0f0; }")
    parts.append("td:first-child, th:first-child { text-align: left; }")
    parts.append("</style></head><body>")
    parts.append(f"<h2>Core AI op frequency — {escape(label)}</h2>")
    parts.append(
        "<table><thead><tr><th>Operation</th><th>Count</th></tr></thead><tbody>"
    )
    for op, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        parts.append(f"<tr><td>{escape(op)}</td><td>{cnt}</td></tr>")
    parts.append(
        f"<tr style='font-weight:bold; border-top:2px solid #333'>"
        f"<td>Total</td><td>{sum(counts.values())}</td></tr>"
    )
    parts.append("</tbody></table>")
    parts.append(f"<p>{len(counts)} unique ops</p>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def print_composites(
    composites: dict[str, int],
    details: dict[str, dict[str, int]],
) -> None:
    """Print composite op frequency tables — one per composite type."""
    if not composites:
        return
    for name, instances in sorted(composites.items(), key=lambda x: x[1], reverse=True):
        inner = details.get(name, {})
        print(f"\n  composite.{name}  ({instances} instances)")
        if inner:
            max_len = max(len(op) for op in inner)
            print(f"  {'─' * max_len}  {'─' * 5}")
            for op, cnt in sorted(inner.items(), key=lambda x: x[1], reverse=True):
                print(f"    {op.ljust(max_len)}  {cnt}")
            total = sum(inner.values())
            per_instance = total // instances if instances else total
            print(
                f"    {''.ljust(max_len)}  {total} total ({per_instance} per instance)"
            )


def format_composites_txt(
    composites: dict[str, int],
    details: dict[str, dict[str, int]],
) -> str:
    """Return plain-text composite detail tables."""
    if not composites:
        return ""
    lines: list[str] = []
    for name, instances in sorted(composites.items(), key=lambda x: x[1], reverse=True):
        inner = details.get(name, {})
        lines.append(f"\n  composite.{name}  ({instances} instances)")
        if inner:
            max_len = max(len(op) for op in inner)
            lines.append(f"  {'─' * max_len}  {'─' * 5}")
            for op, cnt in sorted(inner.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    {op.ljust(max_len)}  {cnt}")
            total = sum(inner.values())
            per_instance = total // instances if instances else total
            lines.append(
                f"    {''.ljust(max_len)}  {total} total ({per_instance} per instance)"
            )
    return "\n".join(lines) + "\n"


def format_composites_html(
    composites: dict[str, int],
    details: dict[str, dict[str, int]],
) -> str:
    """Return HTML composite detail tables (no full page wrapper)."""
    from html import escape

    if not composites:
        return ""
    parts: list[str] = []
    for name, instances in sorted(composites.items(), key=lambda x: x[1], reverse=True):
        inner = details.get(name, {})
        parts.append(f"<h3>composite.{escape(name)} ({instances} instances)</h3>")
        if inner:
            parts.append(
                "<table><thead><tr><th>Operation</th><th>Count</th></tr></thead><tbody>"
            )
            for op, cnt in sorted(inner.items(), key=lambda x: x[1], reverse=True):
                parts.append(f"<tr><td>{escape(op)}</td><td>{cnt}</td></tr>")
            total = sum(inner.values())
            per_instance = total // instances if instances else total
            parts.append(
                f"<tr style='font-weight:bold; border-top:2px solid #333'>"
                f"<td>Total</td><td>{total} ({per_instance} per instance)</td></tr>"
            )
            parts.append("</tbody></table>")
    return "\n".join(parts) + "\n"


def print_composites_comparison(
    composites_a: dict[str, int],
    details_a: dict[str, dict[str, int]],
    composites_b: dict[str, int],
    details_b: dict[str, dict[str, int]],
) -> None:
    """Print per-composite comparison tables (Model 1 vs Model 2 with delta)."""
    RED_BG = "\033[48;2;50;10;10m\033[97m"
    GREEN_BG = "\033[48;2;10;40;10m\033[97m"
    RESET = "\033[0m"

    all_names = sorted(
        set(composites_a) | set(composites_b),
        key=lambda n: max(composites_a.get(n, 0), composites_b.get(n, 0)),
        reverse=True,
    )
    for name in all_names:
        inst_a = composites_a.get(name, 0)
        inst_b = composites_b.get(name, 0)
        inst_delta = inst_b - inst_a
        inst_str = f"{inst_a} vs {inst_b}"
        if inst_delta != 0:
            inst_str += f", {inst_delta:+d}"
        print(f"\n  composite.{name}  ({inst_str} instances)")

        inner_a = details_a.get(name, {})
        inner_b = details_b.get(name, {})
        all_ops = sorted(
            set(inner_a) | set(inner_b),
            key=lambda op: max(inner_a.get(op, 0), inner_b.get(op, 0)),
            reverse=True,
        )
        if not all_ops:
            continue
        max_len = max(len(op) for op in all_ops)
        print(
            f"    {'Operation'.ljust(max_len)}  {'Model 1':>8}  {'Model 2':>8}  {'Delta':>8}"
        )
        print(f"    {'─' * max_len}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
        total_a = total_b = 0
        for op in all_ops:
            a = inner_a.get(op, 0)
            b = inner_b.get(op, 0)
            total_a += a
            total_b += b
            delta = b - a
            delta_str = f"{delta:+d}" if delta != 0 else ""
            marker = " *" if delta != 0 else ""
            line = f"    {op.ljust(max_len)}  {a:>8}  {b:>8}  {delta_str:>8}{marker}"
            if delta < 0:
                line = f"{RED_BG}{line}{RESET}"
            elif delta > 0:
                line = f"{GREEN_BG}{line}{RESET}"
            print(line)
        print(f"    {'─' * max_len}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
        total_delta = total_b - total_a
        delta_str = f"{total_delta:+d}" if total_delta != 0 else "0"
        print(
            f"    {'Total'.ljust(max_len)}  {total_a:>8}  {total_b:>8}  {delta_str:>8}"
        )


def format_composites_comparison_txt(
    composites_a: dict[str, int],
    details_a: dict[str, dict[str, int]],
    composites_b: dict[str, int],
    details_b: dict[str, dict[str, int]],
) -> str:
    """Return plain-text per-composite comparison tables."""
    all_names = sorted(
        set(composites_a) | set(composites_b),
        key=lambda n: max(composites_a.get(n, 0), composites_b.get(n, 0)),
        reverse=True,
    )
    if not all_names:
        return ""
    lines: list[str] = []
    for name in all_names:
        inst_a = composites_a.get(name, 0)
        inst_b = composites_b.get(name, 0)
        inst_delta = inst_b - inst_a
        inst_str = f"{inst_a} vs {inst_b}"
        if inst_delta != 0:
            inst_str += f", {inst_delta:+d}"
        lines.append(f"\n  composite.{name}  ({inst_str} instances)")

        inner_a = details_a.get(name, {})
        inner_b = details_b.get(name, {})
        all_ops = sorted(
            set(inner_a) | set(inner_b),
            key=lambda op: max(inner_a.get(op, 0), inner_b.get(op, 0)),
            reverse=True,
        )
        if not all_ops:
            continue
        max_len = max(len(op) for op in all_ops)
        lines.append(
            f"    {'Operation'.ljust(max_len)}  {'Model 1':>8}  {'Model 2':>8}  {'Delta':>8}"
        )
        lines.append(f"    {'─' * max_len}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
        total_a = total_b = 0
        for op in all_ops:
            a = inner_a.get(op, 0)
            b = inner_b.get(op, 0)
            total_a += a
            total_b += b
            delta = b - a
            delta_str = f"{delta:+d}" if delta != 0 else ""
            marker = " *" if delta != 0 else ""
            lines.append(
                f"    {op.ljust(max_len)}  {a:>8}  {b:>8}  {delta_str:>8}{marker}"
            )
        lines.append(f"    {'─' * max_len}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
        total_delta = total_b - total_a
        delta_str = f"{total_delta:+d}" if total_delta != 0 else "0"
        lines.append(
            f"    {'Total'.ljust(max_len)}  {total_a:>8}  {total_b:>8}  {delta_str:>8}"
        )
    return "\n".join(lines) + "\n"


def format_composites_comparison_html(
    composites_a: dict[str, int],
    details_a: dict[str, dict[str, int]],
    composites_b: dict[str, int],
    details_b: dict[str, dict[str, int]],
) -> str:
    """Return HTML per-composite comparison tables (no full page wrapper)."""
    from html import escape

    all_names = sorted(
        set(composites_a) | set(composites_b),
        key=lambda n: max(composites_a.get(n, 0), composites_b.get(n, 0)),
        reverse=True,
    )
    if not all_names:
        return ""
    parts: list[str] = []
    for name in all_names:
        inst_a = composites_a.get(name, 0)
        inst_b = composites_b.get(name, 0)
        inst_delta = inst_b - inst_a
        inst_str = f"{inst_a} vs {inst_b}"
        if inst_delta != 0:
            inst_str += f", {inst_delta:+d}"
        parts.append(f"<h3>composite.{escape(name)} ({inst_str} instances)</h3>")

        inner_a = details_a.get(name, {})
        inner_b = details_b.get(name, {})
        all_ops = sorted(
            set(inner_a) | set(inner_b),
            key=lambda op: max(inner_a.get(op, 0), inner_b.get(op, 0)),
            reverse=True,
        )
        if not all_ops:
            continue
        parts.append(
            "<table><thead><tr>"
            "<th>Operation</th><th>Model 1</th><th>Model 2</th><th>Delta</th>"
            "</tr></thead><tbody>"
        )
        total_a = total_b = 0
        for op in all_ops:
            a = inner_a.get(op, 0)
            b = inner_b.get(op, 0)
            total_a += a
            total_b += b
            delta = b - a
            delta_str = f"{delta:+d}" if delta != 0 else ""
            css = ""
            if delta > 0:
                css = ' class="added"'
            elif delta < 0:
                css = ' class="removed"'
            parts.append(
                f"<tr{css}><td>{escape(op)}</td><td>{a}</td>"
                f"<td>{b}</td><td>{delta_str}</td></tr>"
            )
        total_delta = total_b - total_a
        delta_str = f"{total_delta:+d}" if total_delta != 0 else "0"
        parts.append(
            f"<tr style='font-weight:bold; border-top:2px solid #333'>"
            f"<td>Total</td><td>{total_a}</td><td>{total_b}</td>"
            f"<td>{delta_str}</td></tr>"
        )
        parts.append("</tbody></table>")
    return "\n".join(parts) + "\n"


def write_output(content: str, path: Path) -> None:
    """Write content to the given path, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Output written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="freqop",
        description=(
            "Compare the frequency of operations in Core AI programs.\n\n"
            "Walks the operation tree and counts coreai.* dialect ops.\n"
            "AIModel assets (.aimodel) are loaded via AIProgram which\n"
            "automatically converts the serialized form to the coreai dialect.\n\n"
            "Composite ops (coreai.graph with a composite_decl attribute) are\n"
            "reported separately as composite.<name> (e.g. composite.layer_norm)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s model.aimodel                        Count ops in a single model\n"
            "  %(prog)s model_a.aimodel model_b.aimodel      Compare two models\n"
            "  %(prog)s --plot model.aimodel                  Count ops and show histogram\n"
            "  %(prog)s --plot model_a.aimodel model_b.aimodel  Compare with grouped bar chart"
        ),
    )
    parser.add_argument(
        "file_a",
        type=Path,
        metavar="FILE",
        help="AIModel asset to analyze (.aimodel)",
    )
    parser.add_argument(
        "file_b",
        type=Path,
        nargs="?",
        default=None,
        metavar="FILE",
        help="optional second AIModel asset to compare against",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="open a matplotlib histogram (grouped bar chart for two-file mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="write output to a file instead of stdout (.txt for plain text, .html for HTML)",
    )
    parser.add_argument(
        "--collapse-composites",
        action="store_true",
        help=(
            "treat each composite graph as a single op instead of counting "
            "its inner ops individually; shows a separate composite frequency table"
        ),
    )
    args = parser.parse_args()

    collapse = args.collapse_composites

    program_a = load_program(args.file_a)
    counts_a, composites_a, details_a = count_operations(
        program_a, collapse_composites=collapse
    )

    if args.file_b is None:
        if args.output:
            fmt = args.output.suffix.lower()
            if fmt == ".html":
                content = format_single_html(counts_a, label=str(args.file_a))
                if composites_a:
                    content = content.replace(
                        "</body></html>",
                        format_composites_html(composites_a, details_a)
                        + "</body></html>",
                    )
            else:
                content = format_single_txt(counts_a, label=str(args.file_a))
                if composites_a:
                    content += "\n" + format_composites_txt(composites_a, details_a)
            write_output(content, args.output)
        else:
            print_single(counts_a, label=args.file_a.name)
            if composites_a:
                print_composites(composites_a, details_a)
        if args.plot:
            plot_single(counts_a, label=str(args.file_a))
    else:
        program_b = load_program(args.file_b)
        counts_b, composites_b, details_b = count_operations(
            program_b, collapse_composites=collapse
        )
        label_a = str(args.file_a)
        label_b = str(args.file_b)
        if args.output:
            fmt = args.output.suffix.lower()
            if fmt == ".html":
                content = format_comparison_html(
                    counts_a, counts_b, label_a=label_a, label_b=label_b
                )
                comp_html = format_composites_comparison_html(
                    composites_a, details_a, composites_b, details_b
                )
                if comp_html:
                    content = content.replace(
                        "</body></html>", comp_html + "</body></html>"
                    )
            else:
                content = format_comparison_txt(
                    counts_a, counts_b, label_a=label_a, label_b=label_b
                )
                comp_txt = format_composites_comparison_txt(
                    composites_a, details_a, composites_b, details_b
                )
                if comp_txt:
                    content += "\n" + comp_txt
            write_output(content, args.output)
        else:
            print_comparison(counts_a, counts_b, label_a=label_a, label_b=label_b)
            print_composites_comparison(
                composites_a, details_a, composites_b, details_b
            )
        if args.plot:
            plot_comparison(
                counts_a,
                counts_b,
                label_a=label_a,
                label_b=label_b,
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
