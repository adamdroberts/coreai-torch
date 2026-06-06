# freqop

Compare the frequency of operations in Core AI programs.

Walks the operation tree and counts `coreai.*` ops. AIModel asset directories (`.aimodel`) are automatically loaded via `AIModelAsset.load(...).program`, which converts the serialized form to the `coreai` dialect.

## Help

```
usage: freqop [-h] [--plot] FILE [FILE]

Compare the frequency of operations in Core AI programs.

Walks the operation tree and counts coreai.* dialect ops.
AIModel assets (.aimodel) are loaded via AIModelAsset which
automatically converts the serialized form to the coreai dialect.

Composite ops (coreai.graph with a composite_decl attribute) are
reported separately as composite.<name> (e.g. composite.layer_norm).

positional arguments:
  FILE        AIModel asset to analyze (.aimodel)
  FILE        optional second AIModel asset to compare against

options:
  -h, --help  show this help message and exit
  --plot      open a matplotlib histogram (grouped bar chart for two-file
              mode)

examples:
  freqop model.aimodel                        Count ops in a single model
  freqop model_a.aimodel model_b.aimodel      Compare two models
  freqop --plot model.aimodel                  Count ops and show histogram
  freqop --plot model_a.aimodel model_b.aimodel  Compare with grouped bar chart
```

## Example

### Single file

Results are sorted by count (descending).

```
  Operation                               Count
  ──────────────────────────────────────  ─────
  constant                                323
  decomposable.broadcasting_add           163
  transpose                               134
  mul                                     64
  composite.layer_norm                    26
  composite.scaled_dot_product_attention  12
  graph                                   1

  Total: 1130 ops  (21 unique)
```

### Two-file comparison

```
  Operation                                Model 1   Model 2     Delta
  ──────────────────────────────────────  ────────  ────────  ────────
  constant                                     323       280       -43 *
  decomposable.broadcasting_add                163       163
  transpose                                    134       120       -14 *
  composite.layer_norm                          26        26
  ──────────────────────────────────────  ────────  ────────  ────────
  Total                                       1130      1073       -57

  Unique ops: 21 vs 20
  2 op(s) differ (marked with *).

  Model 1: path/to/model_a.aimodel
  Model 2: path/to/model_b.aimodel
```

Operations that differ between the two programs are marked with `*`.
Full paths are printed at the bottom for reference.

### Histogram (`--plot`)

Pass `--plot` to open a matplotlib bar chart. For two-file mode, a grouped horizontal bar chart is shown with full paths in the legend.

## Dependencies

- `coreai` — for loading AIModel assets via `AIModelAsset`
- `matplotlib` — only required when using `--plot`
