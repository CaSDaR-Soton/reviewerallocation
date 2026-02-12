
import csv
import math
import random
import re
import os
from pathlib import Path
from typing import List, Set, Dict, Optional

import pandas as pd

print("=== All imports successful ===")

APPLICANTS_FILE = "20260123_Applicants_v1.csv"
REVIEWERS_FILE  = "20260123_reviewers_v1.csv"
OUTPUT_FILE     = "20260101_random_allocation_no_conflicts.csv"

ASSIGNMENTS_PER_APPLICATION = 3
SEED = 42

# NEW: per-reviewer capacity cap
MAX_ASSIGNMENTS_PER_REVIEWER = 5
MIN_ASSIGNMENTS_PER_REVIEWER = 3

EXCLUSION_COL_CANDIDATES = [
    "Reviewer 1", "Reviewer 2", "Reviewer 3"]

ID_PATTERN_INT_OR_FLOAT = re.compile(r"^\s*(\d+)(?:\.0+)?\s*$", re.IGNORECASE)

def normalize_id(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        if val.is_integer():
            return int(val)
        return None
    s = str(val).strip()
    if not s:
        return None
    s_no_commas = s.replace(",", "")
    m = ID_PATTERN_INT_OR_FLOAT.match(s_no_commas)
    if m:
        return int(m.group(1))
    try:
        f = float(s_no_commas)
        if f.is_integer():
            return int(f)
    except ValueError:
        pass
    return None

def load_reviewer_ids(reviewers_path: Path) -> List[int]:
    ids: List[int] = []
    with reviewers_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            raw = row[0]
            rid = normalize_id(raw)
            if rid is None:
                continue
            ids.append(rid)
    seen = set()
    uniq = []
    for rid in ids:
        if rid not in seen:
            uniq.append(rid)
            seen.add(rid)
    return uniq

def get_exclusion_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in EXCLUSION_COL_CANDIDATES if c in df.columns]

def build_exclusion_set(row: pd.Series, exclusion_cols: List[str]) -> Set[int]:
    excluded: Set[int] = set()
    for col in exclusion_cols:
        val = row.get(col)
        rid = normalize_id(val)
        if rid is not None:
            excluded.add(rid)
    return excluded

def choose_reviewers(eligible_pool: List[int], k: int) -> List[int]:
    if len(eligible_pool) <= k:
        return random.sample(eligible_pool, len(eligible_pool)) if eligible_pool else []
    return random.sample(eligible_pool, k)


# --- New helper: min/max aware picking with soft-min priority ---
def choose_reviewers_with_minmax(
    eligible_pool: List[int],
    k: int,
    loads: Dict[int, int],
    min_load: Optional[int],
    max_load: Optional[int],
) -> List[int]:
    """
    Chooses up to k reviewers from eligible_pool such that:
      - Reviewers at/over max_load are excluded.
      - Reviewers currently below min_load are prioritized.
      - Randomness is preserved within ties.
    This is a 'soft' min: if not enough under-min reviewers exist, we fill from the remaining pool.
    """
    # 1) Enforce max cap
    if max_load is not None:
        pool = [rid for rid in eligible_pool if loads.get(rid, 0) < max_load]
    else:
        pool = list(eligible_pool)

    if not pool:
        return []

    if min_load is None or min_load <= 0:
        # No minimum required: simple random sample from the capped pool
        return random.sample(pool, min(k, len(pool)))

    # 2) Split pool by whether reviewers are under the min target
    under_min = [rid for rid in pool if loads.get(rid, 0) < min_load]
    at_or_above_min = [rid for rid in pool if loads.get(rid, 0) >= min_load]

    # 3) Randomize both groups (seeded globally)
    random.shuffle(under_min)
    random.shuffle(at_or_above_min)

    # 4) Pick from under-min first, then fill from the rest
    picks: List[int] = []
    need = k

    take = min(need, len(under_min))
    if take > 0:
        picks.extend(under_min[:take])
        need -= take

    if need > 0 and at_or_above_min:
        fill_take = min(need, len(at_or_above_min))
        picks.extend(at_or_above_min[:fill_take])

    return picks

def main():
    print("=== STARTING MAIN ===")
    if SEED is not None:
        random.seed(SEED)

    applicants_path = Path(APPLICANTS_FILE)
    reviewers_path  = Path(REVIEWERS_FILE)

    df = pd.read_csv(applicants_path)
    if "Application ID" not in df.columns:
        raise ValueError("Applicants CSV must contain an 'Application ID' column.")
    
    df = pd.read_csv(applicants_path)
    print(f"CSV loaded. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    if "Application ID" not in df.columns:
        raise ValueError("Applicants CSV must contain an 'Application ID' column.")

    exclusion_cols = get_exclusion_cols(df)
    if not exclusion_cols:
        print("WARNING: No exclusion columns found; only global random allocation will be performed.")

    reviewer_ids = load_reviewer_ids(reviewers_path)
    if not reviewer_ids:
        raise ValueError("No reviewer IDs found in reviewers CSV.")

    global_reviewers = reviewer_ids[:]
    random.shuffle(global_reviewers)

    # NEW: track loads for capacity enforcement
    loads: Dict[int, int] = {rid: 0 for rid in global_reviewers}

    output_rows: List[Dict[str, object]] = []
    warnings: List[str] = []

    for _, row in df.iterrows():
        app_id = row.get("Application ID")
        print(f"Processing application: {app_id}")  # ← Add this debug line

        excluded = build_exclusion_set(row, exclusion_cols)
        eligible = [rid for rid in global_reviewers if rid not in excluded]

        picks = choose_reviewers_with_minmax(
            eligible_pool=eligible,
            k=ASSIGNMENTS_PER_APPLICATION,
            loads=loads,
            min_load=MIN_ASSIGNMENTS_PER_REVIEWER,
            max_load=MAX_ASSIGNMENTS_PER_REVIEWER
        )

        if len(picks) < ASSIGNMENTS_PER_APPLICATION:
            warnings.append(
                f"Application {app_id}: only {len(picks)} eligible reviewers "
                f"(needed {ASSIGNMENTS_PER_APPLICATION})."
            )

        if set(picks) & excluded:
            raise RuntimeError(
                f"Conflict detected for Application {app_id}: assigned excluded reviewers {set(picks) & excluded}"
            )

        # NEW: update loads for selected reviewers
        for rid in picks:
            loads[rid] = loads.get(rid, 0) + 1

        out = {"Application ID": app_id}
        for i in range(ASSIGNMENTS_PER_APPLICATION):
            out[f"Assigned Reviewer {i+1}"] = picks[i] if i < len(picks) else ""
        out["Excluded Count"] = len(excluded)
        out["Eligible Pool Size"] = len(eligible)
        out["Used Fallback"] = len(picks) < ASSIGNMENTS_PER_APPLICATION
        output_rows.append(out)

    pd.DataFrame(output_rows).to_csv(OUTPUT_FILE, index=False)

    print(f"Random allocation complete (excluded reviewers never assigned).")
    print("\nReviewer Load Distribution:")
    for rid, count in sorted(loads.items(), key=lambda x: x[1], reverse=True):
        print(f"  Reviewer {rid}: {count} assignments")
    print(f"Output file: {Path(OUTPUT_FILE).resolve()}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(" -", w)

if __name__ == "__main__":

    main()
