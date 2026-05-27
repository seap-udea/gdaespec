# Temporary Notebook Review Rules

This temporary note defines the cleanup rules for notebooks inside
`Earth_like_Atmosphere`. Delete it once the notebook review is finished.

## Scope

- Review only notebooks under `Earth_like_Atmosphere`.
- Keep figures and existing scientific outputs untouched unless explicitly asked.
- Do not change physical assumptions, formulas, fitted quantities, filenames, or
  generated data products unless the change is explicitly discussed first.

## Style

- Use English for markdown, comments, error messages, plot labels, and newly
  added text.
- Treat notebooks as scientific workflows, not as production libraries.
- Prefer readable, direct code over broad defensive infrastructure.
- Keep code cells focused on one purpose. Split very large cells when it makes
  the workflow easier to scan.
- Use short markdown cells to explain what each section does and why it exists.

## Functions and Comments

- Add concise docstrings only to functions that encapsulate reusable logic.
- Keep docstrings short: what the function does, key inputs, and the returned
  result are usually enough.
- Add comments only for non-obvious scientific or computational choices, such
  as interpolation, contamination formulas, physical assumptions, or grid
  conventions.
- Avoid comments that merely restate the code.

## Validation

- Keep only minimal checks that catch likely workflow mistakes:
  - missing required local files,
  - unexpected array dimensions,
  - invalid physical ranges or coverage fractions,
  - requested values outside the available model grid.
- Avoid excessive robustness for impossible or out-of-scope cases.
- After editing a notebook, validate:
  - the notebook JSON can be parsed,
  - code cells compile,
  - no Spanish text remains in markdown, comments, or messages,
  - when feasible, regenerated in-memory results match existing products without
    overwriting files.
