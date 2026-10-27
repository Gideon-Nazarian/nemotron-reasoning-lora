"""Print schema + small samples + aggregate stats for competition data.

Per the project rules-discipline (only metadata and small samples are shown),
this script intentionally limits what it shows: schema, dtypes, row counts,
length distributions, and 2 truncated examples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
TRUNCATE = 200  # max chars per cell when previewing


def inspect_csv(path: Path) -> None:
    print(f"\n{'='*70}\n{path.name}\n{'='*70}")
    df = pd.read_csv(path)

    print(f"Shape: {df.shape}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    print(f"\nMissing values: {df.isna().sum().to_dict()}")

    # Length distributions for string columns
    print(f"\nString-column length stats (chars):")
    for col in df.columns:
        if df[col].dtype == object:
            lens = df[col].astype(str).str.len()
            print(
                f"  {col}: min={lens.min()}, p50={int(lens.median())}, "
                f"p95={int(lens.quantile(0.95))}, max={lens.max()}, mean={int(lens.mean())}"
            )

    # 2 truncated samples (first row and last row only — minimize data exposure)
    print(f"\nFirst row (truncated to {TRUNCATE} chars per cell):")
    for col, val in df.iloc[0].items():
        s = str(val)
        if len(s) > TRUNCATE:
            s = s[:TRUNCATE] + f"... [+{len(str(val)) - TRUNCATE} chars]"
        print(f"  {col}: {s!r}")

    if len(df) > 1:
        print(f"\nLast row (truncated to {TRUNCATE} chars per cell):")
        for col, val in df.iloc[-1].items():
            s = str(val)
            if len(s) > TRUNCATE:
                s = s[:TRUNCATE] + f"... [+{len(str(val)) - TRUNCATE} chars]"
            print(f"  {col}: {s!r}")


def main() -> int:
    if not DATA_DIR.exists():
        print(f"data/ not found at {DATA_DIR}", file=sys.stderr)
        return 1

    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSV files in {DATA_DIR}", file=sys.stderr)
        return 1

    for path in csvs:
        inspect_csv(path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
