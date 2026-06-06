# graphdiff

Structural graph diff between two Core AI programs.

Compares the graph structure of two programs and reports added, removed, and reordered operations. AIModel asset directories (`.aimodel`) are automatically loaded via `AIModelAsset.load(...).program`, which converts the serialized form to the `coreai` dialect.

## Help

```
usage: graphdiff [-h] [--entry-point NAME] [--max-items N] [--output FILE] SOURCE TARGET

Structural graph diff between two Core AI programs.

Compares the graph structure of two programs and reports
added, removed, and reordered operations. AIModel assets
(.aimodel) are loaded via AIModelAsset which automatically
converts the serialized form to the coreai dialect.

positional arguments:
  SOURCE              source AIModel asset (.aimodel)
  TARGET              target AIModel asset to compare against (.aimodel)

options:
  -h, --help          show this help message and exit
  --entry-point NAME  coreai.graph entry point to compare (default: all graphs)
  --max-items N       limit the number of items shown in the diff table
  --output FILE       write output to FILE (.html for styled HTML, otherwise plain text)

examples:
  graphdiff model_a.aimodel model_b.aimodel
  graphdiff --entry-point main model_a.aimodel model_b.aimodel
  graphdiff --max-items 50 model_a.aimodel model_b.aimodel
  graphdiff --output diff.html model_a.aimodel model_b.aimodel
```

## Usage

### Basic comparison

By default, graphdiff diffs each `coreai.graph` entry point separately. The `main` graph is diffed first, then composite sub-graphs (e.g., SDPA, layer\_norm) are matched via their `coreai.invoke` call sites in `main` and diffed individually.

```bash
python tools/graphdiff/graphdiff.py model_a.aimodel model_b.aimodel
```

### Single entry point

Compare only a specific `coreai.graph` entry point:

```bash
python tools/graphdiff/graphdiff.py --entry-point main model_a.aimodel model_b.aimodel
```

### Limit output

Limit the number of items shown in the diff table:

```bash
python tools/graphdiff/graphdiff.py --max-items 50 model_a.aimodel model_b.aimodel
```

### Save output to file

Save as HTML (with color highlighting, viewable in a browser or VS Code):

```bash
python tools/graphdiff/graphdiff.py --output diff.html model_a.aimodel model_b.aimodel
```

Save as plain text (ANSI codes stripped):

```bash
python tools/graphdiff/graphdiff.py --output diff.txt model_a.aimodel model_b.aimodel
```

## Composite-aware diffing

When no `--entry-point` is specified, the tool performs per-graph diffing:

1. **Diffs `main` vs `main`** — compares the top-level graphs.
2. **Matches composite sub-graphs** — finds paired `coreai.invoke` ops in the main diff and uses the callee symbols to match composite graphs across the two programs (e.g., source `@sdpa_abc123` with target `@sdpa_def456`).
3. **Diffs each matched composite** — reports a separate diff section per composite.
4. **Reports unmatched composites** — composites present in only one program are reported as added or removed.

This avoids misleading cross-matching that occurs when ops from different sub-graphs are flattened into a single diff.

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | All graphs are isomorphic (no structural differences) |
| 1    | One or more graphs differ structurally |
| 2    | Input error (file not found, invalid arguments) |

The exit code makes `graphdiff` useful in scripts and CI pipelines — e.g., to gate on whether a model change altered the graph structure.

## Dependencies

- `coreai` — for loading AIModel assets via `AIModelAsset`
- `networkx` — for graph isomorphism algorithms

All graph-diff logic is self-contained in `graphdiff.py` (no dependency on the upstream `_graph_diff` implementation).
