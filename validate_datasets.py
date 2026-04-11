"""
Dataset Validation Script
Checks both MIROC6 and ERA5 CSVs against what data_loader.py expects.
"""
import pandas as pd
import numpy as np
import sys

MIROC6_PATH = "MIROC6_UAE_Spatial_Input_1950_2014.csv"
ERA5_PATH   = r"ERA5\UAE_ERA5_Spatial_Baseline_1950_2014.csv"

REQUIRED_COLS = ['Date', 'Lat', 'Lon', 'T_avg', 'PCP', 'AP', 'RH', 'WS']
FEATURES      = ['T_avg', 'PCP', 'AP', 'RH', 'WS']

MIROC6_GRID = 3   # expected 3x3 = 9 rows per date
ERA5_GRID   = 17  # expected 17x17 = 289 rows per date

all_ok = True

def fail(msg):
    global all_ok
    all_ok = False
    print(f"  [FAIL] {msg}")

def ok(msg):
    print(f"  [ OK ] {msg}")

def warn(msg):
    print(f"  [WARN] {msg}")

# ─── MIROC6 ───────────────────────────────────────────────
print("=" * 70)
print(f"VALIDATING MIROC6: {MIROC6_PATH}")
print("=" * 70)

try:
    df_m = pd.read_csv(MIROC6_PATH, nrows=5)
    ok(f"File readable. Columns: {list(df_m.columns)}")
except Exception as e:
    fail(f"Cannot read file: {e}"); sys.exit(1)

# Column check
missing = [c for c in REQUIRED_COLS if c not in df_m.columns]
if missing:
    fail(f"Missing columns: {missing}")
else:
    ok("All required columns present")

# Load full file
print("  Loading full file (this may take a moment)...")
df_m = pd.read_csv(MIROC6_PATH)
df_m['Date'] = pd.to_datetime(df_m['Date']).dt.date

print(f"  Total rows: {len(df_m):,}")
print(f"  Date range: {df_m['Date'].min()} --> {df_m['Date'].max()}")

# Grid consistency
dates_m = df_m['Date'].unique()
rows_per_date_m = df_m.groupby('Date').size()
expected_m = MIROC6_GRID * MIROC6_GRID  # 9

bad_dates_m = rows_per_date_m[rows_per_date_m != expected_m]
if len(bad_dates_m) == 0:
    ok(f"Every date has exactly {expected_m} rows (consistent {MIROC6_GRID}x{MIROC6_GRID} grid)")
else:
    fail(f"{len(bad_dates_m)} dates do NOT have {expected_m} rows")
    print(f"    Value counts of rows-per-date:\n{rows_per_date_m.value_counts().to_string()}")
    print(f"    First few bad dates: {list(bad_dates_m.index[:5])}")

# Unique lat/lon
ulat_m = sorted(df_m['Lat'].unique())
ulon_m = sorted(df_m['Lon'].unique())
print(f"  Unique Lats ({len(ulat_m)}): {ulat_m}")
print(f"  Unique Lons ({len(ulon_m)}): {ulon_m}")
if len(ulat_m) == MIROC6_GRID and len(ulon_m) == MIROC6_GRID:
    ok(f"Grid dimensions match expected {MIROC6_GRID}x{MIROC6_GRID}")
else:
    fail(f"Expected {MIROC6_GRID} lats and {MIROC6_GRID} lons, got {len(ulat_m)} lats and {len(ulon_m)} lons")

# NaN / Inf
for f in FEATURES:
    n_nan = df_m[f].isna().sum()
    n_inf = np.isinf(df_m[f].values).sum() if df_m[f].dtype in [np.float64, np.float32] else 0
    if n_nan > 0: fail(f"'{f}' has {n_nan:,} NaN values")
    elif n_inf > 0: fail(f"'{f}' has {n_inf:,} Inf values")
    else: ok(f"'{f}' — no NaN/Inf  (min={df_m[f].min():.3f}, max={df_m[f].max():.3f})")

# ─── ERA5 ─────────────────────────────────────────────────
print()
print("=" * 70)
print(f"VALIDATING ERA5: {ERA5_PATH}")
print("=" * 70)

try:
    df_e = pd.read_csv(ERA5_PATH, nrows=5)
    ok(f"File readable. Columns: {list(df_e.columns)}")
except Exception as e:
    fail(f"Cannot read file: {e}"); sys.exit(1)

missing_e = [c for c in REQUIRED_COLS if c not in df_e.columns]
if missing_e:
    fail(f"Missing columns: {missing_e}")
else:
    ok("All required columns present")

print("  Loading full file (large — may take a minute)...")
df_e = pd.read_csv(ERA5_PATH)
df_e['Date'] = pd.to_datetime(df_e['Date']).dt.date

print(f"  Total rows: {len(df_e):,}")
print(f"  Date range: {df_e['Date'].min()} --> {df_e['Date'].max()}")

# Grid consistency
dates_e = df_e['Date'].unique()
rows_per_date_e = df_e.groupby('Date').size()
expected_e = ERA5_GRID * ERA5_GRID  # 289

bad_dates_e = rows_per_date_e[rows_per_date_e != expected_e]
if len(bad_dates_e) == 0:
    ok(f"Every date has exactly {expected_e} rows (consistent {ERA5_GRID}x{ERA5_GRID} grid)")
else:
    fail(f"{len(bad_dates_e)} dates do NOT have {expected_e} rows")
    print(f"    Value counts of rows-per-date:\n{rows_per_date_e.value_counts().to_string()}")
    print(f"    First few bad dates: {list(bad_dates_e.index[:5])}")

# Unique lat/lon
ulat_e = sorted(df_e['Lat'].unique())
ulon_e = sorted(df_e['Lon'].unique())
print(f"  Unique Lats ({len(ulat_e)}): {ulat_e}")
print(f"  Unique Lons ({len(ulon_e)}): {ulon_e}")
if len(ulat_e) == ERA5_GRID and len(ulon_e) == ERA5_GRID:
    ok(f"Grid dimensions match expected {ERA5_GRID}x{ERA5_GRID}")
else:
    fail(f"Expected {ERA5_GRID} lats and {ERA5_GRID} lons, got {len(ulat_e)} lats and {len(ulon_e)} lons")

# NaN / Inf
for f in FEATURES:
    if f not in df_e.columns:
        continue
    n_nan = df_e[f].isna().sum()
    n_inf = np.isinf(df_e[f].values).sum() if df_e[f].dtype in [np.float64, np.float32] else 0
    if n_nan > 0: fail(f"'{f}' has {n_nan:,} NaN values")
    elif n_inf > 0: fail(f"'{f}' has {n_inf:,} Inf values")
    else: ok(f"'{f}' — no NaN/Inf  (min={df_e[f].min():.3f}, max={df_e[f].max():.3f})")

# ─── CROSS-DATASET ALIGNMENT ─────────────────────────────
print()
print("=" * 70)
print("CROSS-DATASET ALIGNMENT CHECK")
print("=" * 70)

set_m = set(dates_m)
set_e = set(dates_e)
common = sorted(set_m & set_e)
only_m = set_m - set_e
only_e = set_e - set_m

print(f"  MIROC6 dates: {len(set_m):,}")
print(f"  ERA5   dates: {len(set_e):,}")
print(f"  Common dates: {len(common):,}")

if len(only_m) > 0:
    warn(f"{len(only_m)} dates in MIROC6 but NOT in ERA5 (first 5): {sorted(only_m)[:5]}")
if len(only_e) > 0:
    warn(f"{len(only_e)} dates in ERA5 but NOT in MIROC6 (first 5): {sorted(only_e)[:5]}")

if len(common) == 0:
    fail("No overlapping dates between datasets — training is impossible!")
elif len(common) == len(set_m) == len(set_e):
    ok("Perfect 1:1 date alignment between both datasets")
else:
    ok(f"Partial overlap: {len(common):,} usable training dates")

# Sequence feasibility
SEQ_LEN = 14
usable_seqs = max(0, len(common) - SEQ_LEN + 1)
print(f"\n  Usable sliding-window sequences (seq_length={SEQ_LEN}): {usable_seqs:,}")
if usable_seqs == 0:
    fail("Not enough aligned dates to form even one training sequence!")
else:
    ok(f"Sufficient for training ({usable_seqs:,} sequences)")

# ─── RESHAPE SMOKE TEST ──────────────────────────────────
print()
print("=" * 70)
print("RESHAPE SMOKE TEST (data_loader compatibility)")
print("=" * 70)

test_date = common[0]
print(f"  Testing reshape on date: {test_date}")

# MIROC6 reshape
grp_m = df_m[df_m['Date'] == test_date].sort_values(['Lat', 'Lon'])
try:
    for feat in FEATURES:
        mat = grp_m[feat].values.reshape(MIROC6_GRID, MIROC6_GRID)
    ok(f"MIROC6 reshaped to ({MIROC6_GRID},{MIROC6_GRID}) for all features")
except Exception as e:
    fail(f"MIROC6 reshape failed: {e}")

# ERA5 reshape
grp_e = df_e[df_e['Date'] == test_date].sort_values(['Lat', 'Lon'])
try:
    for feat in FEATURES:
        mat = grp_e[feat].values.reshape(ERA5_GRID, ERA5_GRID)
    ok(f"ERA5 reshaped to ({ERA5_GRID},{ERA5_GRID}) for all features")
except Exception as e:
    fail(f"ERA5 reshape failed: {e}")

# ─── SUMMARY ─────────────────────────────────────────────
print()
print("=" * 70)
if all_ok:
    print("RESULT: ALL CHECKS PASSED ✓  — Datasets are ready for training!")
else:
    print("RESULT: SOME CHECKS FAILED ✗  — Fix the issues above before training.")
print("=" * 70)
