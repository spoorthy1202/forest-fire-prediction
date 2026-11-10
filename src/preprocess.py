"""
Dedicated Preprocessing Module for the Extended Algerian Forest Fires Dataset.

This module cleans, normalizes, validates, and analyzes the extended Algerian
forest fires dataset (1,215 rows across 10 geographical region-year groups).

Key tasks performed:
1. Column name normalization (strip whitespace, rename 'Bejaia 2021' to 'Region').
2. Whitespace stripping for all categorical/string values.
3. Classes normalization ('fire' and 'not fire') and binary encoding ('fire' -> 1, 'not fire' -> 0).
4. Region naming standardization (removes inconsistent whitespace, separators, mashed years).
5. DATE parsing (including Excel numeric serial date 44439 -> 2021-08-31) and validation.
6. Duplicate checks (full row and Region+DATE duplicates).
7. Missing value checks and IQR-based numerical outlier detection (reported without silent deletion).
8. Cleaned CSV generation (at dataset/processed/forest_fires_cleaned.csv).
9. Comprehensive data quality report generation (at outputs/reports/data_quality_report.json).
10. Descriptive statistics and class distribution calculations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Default paths
DEFAULT_INPUT_PATH = Path("dataset/The extended Algerian forest fires dataset.xlsx")
DEFAULT_OUTPUT_CSV = Path("dataset/processed/forest_fires_cleaned.csv")
DEFAULT_REPORT_JSON = Path("outputs/reports/data_quality_report.json")
DEFAULT_SHEET_NAME = "DONAT_data_set"

NUMERICAL_COLUMNS = [
    "Temperature",
    "Ws",
    "Rain",
    "RH",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI",
]


def load_raw_data(
    filepath: Path | str = DEFAULT_INPUT_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
) -> pd.DataFrame:
    """Load the raw extended dataset from Excel while preserving the original file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset file not found: {path.resolve()}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df


def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize column names:
    1. Strip leading and trailing whitespace from all column names.
    2. Rename 'Bejaia 2021' to 'Region'.
    """
    df = df.copy()
    original_cols = df.columns.tolist()

    # Strip whitespace
    stripped_cols = [c.strip() for c in original_cols]
    whitespace_stripped = [
        orig for orig, stripped in zip(original_cols, stripped_cols) if orig != stripped
    ]
    df.columns = stripped_cols

    # Rename region column
    renamed = {}
    if "Bejaia 2021" in df.columns:
        df = df.rename(columns={"Bejaia 2021": "Region"})
        renamed["Bejaia 2021"] = "Region"

    meta = {
        "original_columns": original_cols,
        "normalized_columns": df.columns.tolist(),
        "whitespace_stripped_columns": whitespace_stripped,
        "renamed_columns": renamed,
    }
    return df, meta


def strip_categorical_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Strip leading and trailing whitespace from all string/categorical columns."""
    df = df.copy()
    stripped_details = {}

    for col in df.select_dtypes(include=["object"]).columns:
        if col == "DATE":
            continue
        original_series = df[col].astype(str)
        cleaned_series = original_series.str.strip()
        diff_count = (original_series != cleaned_series).sum()
        df[col] = cleaned_series
        stripped_details[col] = int(diff_count)

    return df, {"categorical_whitespace_stripped_counts": stripped_details}


def standardize_region_name(raw_name: Any) -> str:
    """
    Standardize region name:
    - Strips whitespace.
    - Inserts a space before 4-digit years if attached to letters (e.g. Tipaza2017 -> Tipaza 2017).
    - Replaces inconsistent separators (_ and -) with space.
    - Collapses multiple whitespace into a single space.
    - Preserves region name and year identity.
    """
    s = str(raw_name).strip()
    # Insert space before 4-digit year at end if attached to a letter
    s = re.sub(r"([a-zA-Z])(\d{4})$", r"\1 \2", s)
    # Replace underscores and hyphens with spaces
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"[-]+", " ", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def standardize_regions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Standardize Region column and verify region-year groups."""
    df = df.copy()
    if "Region" not in df.columns:
        raise KeyError("Column 'Region' not found. Ensure column normalization has run.")

    raw_unique = df["Region"].unique().tolist()
    mapping = {raw: standardize_region_name(raw) for raw in raw_unique}
    df["Region"] = df["Region"].map(mapping)

    cleaned_counts = df["Region"].value_counts().to_dict()
    num_groups = len(cleaned_counts)

    meta = {
        "raw_region_values": raw_unique,
        "mapping": mapping,
        "num_region_year_groups": num_groups,
        "region_counts": {k: int(v) for k, v in cleaned_counts.items()},
    }
    return df, meta


def normalize_and_encode_classes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize Classes into exactly 'fire' and 'not fire', then encode:
    - 'fire' -> 1
    - 'not fire' -> 0
    """
    df = df.copy()
    if "Classes" not in df.columns:
        raise KeyError("Column 'Classes' not found.")

    raw_counts = df["Classes"].value_counts().to_dict()

    # Normalize strings: lowercase and stripped
    classes_clean = df["Classes"].astype(str).str.strip().str.lower()
    normalized_counts = classes_clean.value_counts().to_dict()

    # Validate distinct values
    unexpected = set(classes_clean.unique()) - {"fire", "not fire"}
    if unexpected:
        raise ValueError(f"Unexpected values in Classes column: {unexpected}")

    # Binary encoding: fire -> 1, not fire -> 0
    mapping = {"fire": 1, "not fire": 0}
    df["Classes"] = classes_clean.map(mapping).astype(int)

    encoded_counts = df["Classes"].value_counts().to_dict()

    meta = {
        "raw_counts": {str(k): int(v) for k, v in raw_counts.items()},
        "normalized_counts": {str(k): int(v) for k, v in normalized_counts.items()},
        "encoded_counts": {str(k): int(v) for k, v in encoded_counts.items()},
        "target_mapping": mapping,
    }
    return df, meta


def parse_single_date(val: Any) -> pd.Timestamp:
    """Parse a single date value handling datetime, Excel serial integers, and strings."""
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        # Excel serial date (origin: 1899-12-30 for 1900 date system)
        return pd.to_datetime(val, unit="D", origin="1899-12-30")
    if isinstance(val, str):
        v = val.strip()
        if v.isdigit():
            return pd.to_datetime(int(v), unit="D", origin="1899-12-30")
        return pd.to_datetime(v)
    return pd.to_datetime(val)


def parse_and_validate_dates(
    df: pd.DataFrame,
    fix_year_mismatches: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse DATE and validate invalid dates.

    Checks:
    - Identifies Excel serial date integers (e.g., row 818 = 44439 -> 2021-08-31).
    - Checks for unparseable (NaT) dates.
    - Checks for year discrepancies between DATE and the Region year.
      (e.g., Setif 2021 contains 5 records from 2025: 2025-09-25, 2025-09-27..30).
    - If fix_year_mismatches is True, harmonizes the date year with the region group year.
    - Formats output DATE as ISO 'YYYY-MM-DD' strings.
    """
    df = df.copy()
    if "DATE" not in df.columns:
        raise KeyError("Column 'DATE' not found.")

    excel_serials = []
    unparseable = []
    year_mismatches = []
    parsed_dates: List[pd.Timestamp] = []

    for idx, (raw_val, reg) in enumerate(zip(df["DATE"], df["Region"])):
        # Detect Excel serial integers
        if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
            excel_serials.append(
                {
                    "row_index": int(idx),
                    "raw_value": int(raw_val) if isinstance(raw_val, int) else float(raw_val),
                    "region": str(reg),
                }
            )

        ts = parse_single_date(raw_val)
        if pd.isna(ts):
            unparseable.append({"row_index": int(idx), "raw_value": str(raw_val), "region": str(reg)})
            parsed_dates.append(ts)
            continue

        # Extract expected 4-digit year from Region
        reg_year_match = re.search(r"\b(19\d\d|20\d\d)\b", str(reg))
        expected_year = int(reg_year_match.group(1)) if reg_year_match else None

        if expected_year and ts.year != expected_year:
            mismatch_info = {
                "row_index": int(idx),
                "raw_date": str(ts.date()),
                "region": str(reg),
                "expected_year": int(expected_year),
                "actual_year": int(ts.year),
            }
            if fix_year_mismatches:
                # Correct date to expected year
                ts_corrected = ts.replace(year=expected_year)
                mismatch_info["resolved_date"] = str(ts_corrected.date())
                mismatch_info["action"] = f"Corrected year from {ts.year} to {expected_year}"
                ts = ts_corrected
            else:
                mismatch_info["action"] = "Preserved as parsed; flagged mismatch"

            year_mismatches.append(mismatch_info)

        parsed_dates.append(ts)

    df["DATE"] = [d.strftime("%Y-%m-%d") if pd.notna(d) else None for d in parsed_dates]

    meta = {
        "total_dates_evaluated": len(df),
        "unparseable_dates_count": len(unparseable),
        "unparseable_dates": unparseable,
        "excel_serials_converted_count": len(excel_serials),
        "excel_serials_converted": excel_serials,
        "year_mismatches_detected_count": len(year_mismatches),
        "year_mismatches": year_mismatches,
        "year_mismatches_harmonized": fix_year_mismatches,
        "date_ranges_by_region": {
            reg: {
                "min": str(grp["DATE"].min()),
                "max": str(grp["DATE"].max()),
                "count": int(len(grp)),
            }
            for reg, grp in df.groupby("Region")
        },
    }
    return df, meta


def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for duplicate rows and duplicate Region+DATE combinations."""
    exact_dups = int(df.duplicated().sum())
    region_date_dups = int(df.duplicated(subset=["Region", "DATE"]).sum()) if {"Region", "DATE"}.issubset(df.columns) else 0

    return {
        "exact_row_duplicates": exact_dups,
        "region_date_duplicates": region_date_dups,
    }


def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Check missing values per column and overall."""
    missing_by_col = df.isnull().sum().to_dict()
    total_missing = int(sum(missing_by_col.values()))

    return {
        "total_missing_values": total_missing,
        "missing_per_column": {k: int(v) for k, v in missing_by_col.items()},
    }


def detect_numerical_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect numerical outliers using the IQR rule (1.5 * IQR).
    All outliers are documented and reported; no observations are deleted.
    """
    outlier_report = {}
    for col in NUMERICAL_COLUMNS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = float(q25 - 1.5 * iqr)
        upper_bound = float(q75 + 1.5 * iqr)

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = int(len(outliers))
        pct = float(round((outlier_count / len(series)) * 100, 2))

        outlier_report[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "q25": round(q25, 2),
            "median": round(float(series.median()), 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": outlier_count,
            "outlier_percentage": pct,
            "action": "reported_only_preserved_in_dataset",
        }

    return outlier_report


def compute_descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute descriptive statistics for all numerical features."""
    stats = {}
    for col in NUMERICAL_COLUMNS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "25%": round(float(s.quantile(0.25)), 4),
            "50%": round(float(s.quantile(0.50)), 4),
            "75%": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
        }
    return stats


def compute_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall and regional class distribution."""
    if "Classes" not in df.columns:
        raise KeyError("Column 'Classes' not found.")

    counts = df["Classes"].value_counts().to_dict()
    total = len(df)
    distribution = {
        "0_not_fire": {
            "count": int(counts.get(0, 0)),
            "percentage": round(float((counts.get(0, 0) / total) * 100), 2),
        },
        "1_fire": {
            "count": int(counts.get(1, 0)),
            "percentage": round(float((counts.get(1, 0) / total) * 100), 2),
        },
    }

    by_region = {}
    if "Region" in df.columns:
        ct = pd.crosstab(df["Region"], df["Classes"])
        for reg in ct.index:
            not_fire_cnt = int(ct.loc[reg, 0]) if 0 in ct.columns else 0
            fire_cnt = int(ct.loc[reg, 1]) if 1 in ct.columns else 0
            reg_total = not_fire_cnt + fire_cnt
            by_region[reg] = {
                "not_fire_0": not_fire_cnt,
                "fire_1": fire_cnt,
                "total": reg_total,
                "fire_rate_percentage": round(float((fire_cnt / reg_total) * 100), 2) if reg_total > 0 else 0.0,
            }

    return {
        "overall": distribution,
        "by_region": by_region,
    }


def preprocess_dataset(
    input_excel_path: Path | str = DEFAULT_INPUT_PATH,
    output_csv_path: Path | str = DEFAULT_OUTPUT_CSV,
    report_json_path: Path | str = DEFAULT_REPORT_JSON,
    fix_year_mismatches: bool = True,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the complete preprocessing pipeline on the extended Algerian forest fires dataset.

    Returns:
        cleaned_df: The fully cleaned and encoded DataFrame (1,215 rows).
        report: Detailed data quality, outlier, validation, and statistical report dictionary.
    """
    input_path = Path(input_excel_path)
    out_csv = Path(output_csv_path)
    out_report = Path(report_json_path)

    # 1. Load data
    df_raw = load_raw_data(input_path)
    raw_shape = [int(df_raw.shape[0]), int(df_raw.shape[1])]

    # 2. Normalize columns (strip whitespace & rename Bejaia 2021 -> Region)
    df, col_meta = normalize_column_names(df_raw)

    # 3. Strip whitespace from all categorical values
    df, cat_meta = strip_categorical_values(df)

    # 4. Standardize region naming
    df, reg_meta = standardize_regions(df)

    # 5. Normalize and encode Classes (fire -> 1, not fire -> 0)
    df, class_meta = normalize_and_encode_classes(df)

    # 6. Parse and validate DATE
    df, date_meta = parse_and_validate_dates(df, fix_year_mismatches=fix_year_mismatches)

    # 7. Check duplicates
    dup_meta = check_duplicates(df)

    # 8. Check missing values
    missing_meta = check_missing_values(df)

    # 9. Numerical outliers (IQR)
    outlier_meta = detect_numerical_outliers(df)

    # 10. Descriptive statistics
    stats_meta = compute_descriptive_statistics(df)

    # 11. Class distribution
    dist_meta = compute_class_distribution(df)

    # Assemble comprehensive report
    report = {
        "dataset_overview": {
            "input_file": str(input_path.as_posix()),
            "sheet_name": DEFAULT_SHEET_NAME,
            "raw_shape": raw_shape,
            "cleaned_shape": [int(df.shape[0]), int(df.shape[1])],
            "row_count_preserved": bool(df.shape[0] == 1215),
            "columns": df.columns.tolist(),
            "fwi_usage_note": "FWI is retained in the cleaned dataset for completeness but MUST NOT be used as a prediction feature during model training to avoid target leakage.",
        },
        "column_normalization": col_meta,
        "categorical_normalization": cat_meta,
        "region_standardization": reg_meta,
        "target_encoding": class_meta,
        "date_validation": date_meta,
        "duplicates": dup_meta,
        "missing_values": missing_meta,
        "numerical_outliers_iqr": outlier_meta,
        "descriptive_statistics": stats_meta,
        "class_distribution": dist_meta,
    }

    # Ensure output directories exist and save files
    if save_outputs:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_report.parent.mkdir(parents=True, exist_ok=True)

        # Save cleaned CSV
        df.to_csv(out_csv, index=False)
        print(f"[OK] Cleaned dataset saved to: {out_csv.resolve()} ({len(df)} rows, {len(df.columns)} columns)")

        # Save data quality report JSON
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[OK] Data quality report saved to: {out_report.resolve()}")

    return df, report


def main() -> None:
    """CLI entry point for running dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description="Dedicated preprocessing module for the Extended Algerian Forest Fires dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to raw extended Excel file",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path for cleaned CSV output",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT_JSON,
        help="Path for data quality report JSON",
    )
    parser.add_argument(
        "--no-fix-year-typos",
        action="store_true",
        help="Disable automatic harmonization of Setif 2021 year-mismatch dates (2025 -> 2021)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Extended Algerian Forest Fires Dataset - Preprocessing")
    print("=" * 70)

    cleaned_df, report = preprocess_dataset(
        input_excel_path=args.input,
        output_csv_path=args.output_csv,
        report_json_path=args.report_json,
        fix_year_mismatches=not args.no_fix_year_typos,
        save_outputs=True,
    )

    print("\n--- Summary Highlights ---")
    print(f"Total Rows Preserved: {len(cleaned_df)} / 1,215")
    print(f"Region-Year Groups: {report['region_standardization']['num_region_year_groups']}")
    print(f"Target Distribution: 0 (not fire) = {report['class_distribution']['overall']['0_not_fire']['count']}, 1 (fire) = {report['class_distribution']['overall']['1_fire']['count']}")
    print(f"Missing Values: {report['missing_values']['total_missing_values']}")
    print(f"Duplicate Rows: {report['duplicates']['exact_row_duplicates']}")
    print(f"Region+Date Duplicates: {report['duplicates']['region_date_duplicates']}")
    print(f"Excel Serial Dates Converted: {report['date_validation']['excel_serials_converted_count']}")
    print(f"Date Year Mismatches Detected: {report['date_validation']['year_mismatches_detected_count']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
