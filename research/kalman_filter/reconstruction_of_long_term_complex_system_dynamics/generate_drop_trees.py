"""
Search for alternative `drop_trees_list` variants for the "Data distortion"
step in dendrochronological_series.ipynb.

A column like "ff201a" / "ff201b" is not two trees, it's two cores/samples
from the SAME physical tree "ff201" -- so they are always kept or dropped
together as one unit.

Goal: pick a subset of tree GROUPS (not individual core columns) to drop
so that
  1) the *median* percentage of remaining (non-NaN) observations per year,
     relative to the full tree set, falls within TARGET_PCT_RANGE (20-25 %)
     -- matching the criterion already used in the notebook (see the
     "Min" / "Median" print cell), and
  2) every single year keeps at least MIN_TREES_PER_YEAR core(s) of data
     (no year is fully wiped out by the distortion).

Usage:
    python generate_drop_trees.py

Prints N_VARIANTS ready-to-paste `drop_trees_list` variants plus their
min/median/mean coverage stats.
"""

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

TREE_COLUMN_RE = re.compile(r"^(ff\d+)[a-z]?$")

DATA_DIR = Path(__file__).parent / "data"
TARGET_PCT_RANGE = (20.0, 25.0)  # desired median % of remaining data per year
MIN_TREES_PER_YEAR = 1  # every year must keep at least this many trees
N_VARIANTS = 5  # how many additional drop lists to produce
MAX_JACCARD_OVERLAP = 0.6  # keep-sets of different variants shouldn't be too similar
MAX_TRIALS = 200_000
RNG_SEED = 42

# Trees already used in the notebook's current `drop_trees_list` (kept set is its complement).
EXISTING_DROP_TREES_LIST = [
    "ff201a",
    "ff201b",
    "ff202a",
    "ff202b",
    "ff203a",
    "ff203b",
    "ff204a",
    "ff204b",
    "ff205a",
    "ff205b",
    "ff207a",
    "ff207b",
    "ff208a",
    "ff208b",
    "ff209a",
    "ff209b",
    "ff210a",
    "ff210b",
    "ff211a",
    "ff211b",
    "ff213a",
    "ff213b",
    "ff216a",
    "ff216b",
    "ff217a",
    "ff217b",
    "ff218a",
    "ff218b",
    "ff220a",
    "ff220b",
    "ff221a",
    "ff221b",
    "ff222a",
    "ff222b",
    "ff224a",
    "ff224b",
    "ff225a",
    "ff225b",
    "ff228a",
    "ff228b",
    "ff229a",
    "ff229b",
    "ff252a",
    "ff252b",
    "ff253a",
    "ff253c",
    "ff254a",
    "ff254b",
    "ff255a",
    "ff255b",
    "ff256a",
    "ff256b",
    "ff257b",
    "ff257c",
    "ff260b",
    "ff260c",
]


def load_full_frame() -> pd.DataFrame:
    df_rwi = pd.read_csv(DATA_DIR / "tree_rwi.csv", index_col="year")
    all_years = np.arange(df_rwi.index.min(), df_rwi.index.max() + 1)
    return df_rwi.reindex(all_years)


def tree_id(column: str) -> str:
    """Map a core column ("ff201a") to its physical tree id ("ff201")."""
    m = TREE_COLUMN_RE.match(column)
    if not m:
        raise ValueError(f"Unexpected column name: {column!r}")
    return m.group(1)


def build_tree_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in columns:
        groups.setdefault(tree_id(c), []).append(c)
    return groups


def columns_for(tree_ids: set, groups: dict[str, list[str]]) -> list[str]:
    return [c for tid in tree_ids for c in groups[tid]]


def pct_stats(df_full: pd.DataFrame, total_by_year: pd.Series, keep_columns: list[str]):
    aug_by_year = (
        df_full[keep_columns].notna().sum(axis=1).reindex(df_full.index, fill_value=0)
    )
    pct = (aug_by_year / total_by_year.replace(0, pd.NA) * 100).fillna(0)
    return aug_by_year.min(), pct.min(), float(np.median(pct)), pct.mean()


def tree_ids_required_for_min_coverage(
    df_full: pd.DataFrame, groups: dict[str, list[str]], min_trees_per_year: int
) -> set:
    """
    Tree ids that MUST be kept because, for at least one year, fewer than
    min_trees_per_year OTHER cores have data -- i.e. dropping them would
    push that year below the minimum, no matter what else is kept.
    """
    present = df_full.notna()
    required = set()
    for _, row in present.iterrows():
        active_columns = row.index[row].tolist()
        if len(active_columns) <= min_trees_per_year:
            required.update(tree_id(c) for c in active_columns)
    return required


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def main():
    df_full = load_full_frame()
    all_columns = df_full.columns.tolist()
    groups = build_tree_groups(all_columns)
    all_tree_ids = list(groups.keys())
    total_by_year = df_full.notna().sum(axis=1)

    existing_drop_ids = {tree_id(c) for c in EXISTING_DROP_TREES_LIST}
    existing_keep_ids = set(all_tree_ids) - existing_drop_ids
    required_ids = tree_ids_required_for_min_coverage(
        df_full, groups, MIN_TREES_PER_YEAR
    )
    print(f"Total trees: {len(all_tree_ids)} ({len(all_columns)} cores)")
    print(
        f"Existing variant keeps {len(existing_keep_ids)} trees "
        f"({100 * len(existing_keep_ids) / len(all_tree_ids):.1f}% of trees)"
    )
    print(
        f"Trees that must stay to keep >= {MIN_TREES_PER_YEAR} core(s)/year: "
        f"{sorted(required_ids)}"
    )
    missing_required = required_ids - existing_keep_ids
    if missing_required:
        print(
            f"WARNING: existing drop_trees_list violates the "
            f"{MIN_TREES_PER_YEAR}-core/year minimum (drops {sorted(missing_required)})"
        )

    rng = random.Random(RNG_SEED)
    found_keep_id_sets: list[set] = [existing_keep_ids]
    results = []

    # Candidate keep-counts: roughly TARGET_PCT_RANGE share of all trees,
    # widened a bit since per-year coverage isn't identical to the tree-count share.
    lo_count = max(len(required_ids), int(len(all_tree_ids) * 0.15))
    hi_count = int(len(all_tree_ids) * 0.32) + 1
    optional_ids = [t for t in all_tree_ids if t not in required_ids]

    trials = 0
    while len(results) < N_VARIANTS and trials < MAX_TRIALS:
        trials += 1
        keep_count = rng.randint(lo_count, hi_count)
        extra_count = max(0, keep_count - len(required_ids))
        keep_ids = required_ids | set(
            rng.sample(optional_ids, min(extra_count, len(optional_ids)))
        )

        if any(
            jaccard(keep_ids, prev) > MAX_JACCARD_OVERLAP for prev in found_keep_id_sets
        ):
            continue

        keep_columns = columns_for(keep_ids, groups)
        min_count, min_pct, median_pct, mean_pct = pct_stats(
            df_full, total_by_year, keep_columns
        )
        if min_count < MIN_TREES_PER_YEAR:
            continue  # a year would be fully wiped out (or below the minimum)
        if TARGET_PCT_RANGE[0] <= median_pct <= TARGET_PCT_RANGE[1]:
            found_keep_id_sets.append(keep_ids)
            results.append((keep_ids, min_pct, median_pct, mean_pct))

    print(
        f"\nSearched {trials} random subsets, found {len(results)}/{N_VARIANTS} variants.\n"
    )

    for i, (keep_ids, min_pct, median_pct, mean_pct) in enumerate(results, start=2):
        drop_list = sorted(columns_for(set(all_tree_ids) - keep_ids, groups))
        print(
            f"# --- variant {i}: kept={len(keep_ids)} trees, "
            f"min={min_pct:.1f}%, median={median_pct:.1f}%, mean={mean_pct:.1f}% ---"
        )
        print(f"drop_trees_list_{i} = [")
        for j in range(0, len(drop_list), 8):
            chunk = ", ".join(f'"{t}"' for t in drop_list[j : j + 8])
            print(f"    {chunk},")
        print("]\n")


if __name__ == "__main__":
    main()
