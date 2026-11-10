"""
Unit and Integration Tests for Preprocessing Module.

Tests:
1. Exact row count preservation (1,215 rows).
2. Column names normalization (stripping leading/trailing whitespace, renaming 'Bejaia 2021' -> 'Region').
3. DATE and Region retention (not dropped).
4. Categorical whitespace stripping.
5. Classes target normalization and binary encoding (0 and 1 only; exact counts: 633 fire, 582 not fire).
6. Region naming standardization and verification of exactly 10 distinct region-year groups.
7. Correct parsing of Excel serial dates (row 818 = 44439 -> 2021-08-31) and date validation.
8. Duplicate row and duplicate region-date verification (0 duplicates).
9. Missing value checks (0 missing values).
10. Preservation of the original Excel file.
11. Verification that outputs exist and have valid structure (forest_fires_cleaned.csv and data_quality_report.json).
"""

import json
import os
import sys
from pathlib import Path
import pytest
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocess import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_REPORT_JSON,
    load_raw_data,
    preprocess_dataset,
    standardize_region_name,
)


@pytest.fixture(scope="module")
def preprocessing_results():
    """Run preprocessing once and provide cleaned df and report for tests."""
    df, report = preprocess_dataset(
        input_excel_path=DEFAULT_INPUT_PATH,
        output_csv_path=DEFAULT_OUTPUT_CSV,
        report_json_path=DEFAULT_REPORT_JSON,
        fix_year_mismatches=True,
        save_outputs=True,
    )
    return df, report


def test_row_count_preserved(preprocessing_results):
    """Test confirming exactly 1,215 rows are preserved."""
    cleaned_df, report = preprocessing_results
    assert len(cleaned_df) == 1215, f"Expected 1,215 rows, found {len(cleaned_df)}"
    assert report["dataset_overview"]["row_count_preserved"] is True
    assert report["dataset_overview"]["cleaned_shape"][0] == 1215


def test_column_names_normalized(preprocessing_results):
    """Test column names normalized, whitespace stripped, and renamed to 'Region'."""
    cleaned_df, report = preprocessing_results
    cols = cleaned_df.columns.tolist()

    # Verify no leading/trailing whitespace in column names
    for col in cols:
        assert col == col.strip(), f"Column name '{col}' contains unstripped whitespace"

    # Verify 'Bejaia 2021' renamed to 'Region'
    assert "Region" in cols, "Column 'Region' missing from cleaned dataframe"
    assert "Bejaia 2021" not in cols, "Raw column 'Bejaia 2021' was not renamed"

    # Verify DATE and Region preserved (not dropped)
    assert "DATE" in cols, "DATE column was dropped"
    assert "Region" in cols, "Region column was dropped"

    # Verify expected column set
    expected_cols = [
        "DATE",
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
        "Classes",
        "Region",
    ]
    assert cols == expected_cols, f"Column mismatch: expected {expected_cols}, got {cols}"


def test_target_classes_encoded_binary(preprocessing_results):
    """Verify target Classes contains ONLY 0 and 1, with exact distribution."""
    cleaned_df, report = preprocessing_results
    unique_classes = set(cleaned_df["Classes"].unique())
    assert unique_classes == {0, 1}, f"Target contains unexpected values: {unique_classes}"

    # Verify exact counts: fire=633, not fire=582
    counts = cleaned_df["Classes"].value_counts().to_dict()
    assert counts[1] == 633, f"Expected 633 fire (1), found {counts.get(1)}"
    assert counts[0] == 582, f"Expected 582 not fire (0), found {counts.get(0)}"
    assert counts[1] + counts[0] == 1215


def test_ten_region_year_groups(preprocessing_results):
    """Verify that there are exactly 10 region-year groups with proper counts."""
    cleaned_df, report = preprocessing_results
    regions = cleaned_df["Region"].unique()

    assert len(regions) == 10, f"Expected exactly 10 region-year groups, found {len(regions)}"
    assert report["region_standardization"]["num_region_year_groups"] == 10

    # Verify counts per region: Khenchela has 117, other 9 groups have 122 each
    counts = cleaned_df["Region"].value_counts().to_dict()
    for reg, count in counts.items():
        if "Khenchela" in reg:
            assert count == 117, f"Expected 117 for {reg}, found {count}"
        else:
            assert count == 122, f"Expected 122 for {reg}, found {count}"

    # Verify region naming has no leading/trailing whitespace or inconsistent separators (_ or -)
    for reg in regions:
        assert reg == reg.strip(), f"Region name '{reg}' has unstripped whitespace"
        assert "_" not in reg, f"Region name '{reg}' contains underscore separator"
        assert "-" not in reg, f"Region name '{reg}' contains hyphen separator"


def test_standardize_region_name_function():
    """Unit test for region name standardizer on all raw dataset variations."""
    raw_samples = {
        "Bejaia 2021": "Bejaia 2021",
        "Blida 2021": "Blida 2021",
        "Bouira 2021 ": "Bouira 2021",
        "Khenchela 2021": "Khenchela 2021",
        "Tipaza2017": "Tipaza 2017",
        "Setif 2021": "Setif 2021",
        "Tiaret 2021": "Tiaret 2021",
        "Bejaia_2012": "Bejaia 2012",
        "Sidi-Bel-Abass2012": "Sidi Bel Abass 2012",
        "Tizi-Ouzou2021": "Tizi Ouzou 2021",
    }
    for raw, expected in raw_samples.items():
        assert standardize_region_name(raw) == expected, f"Failed for {raw}"


def test_date_parsing_and_excel_serial_handling(preprocessing_results):
    """Verify date parsing handles Excel serial date number (row 818 = 44439 -> 2021-08-31)."""
    cleaned_df, report = preprocessing_results

    # Check that row 818 (Tiaret 2021 serial 44439) was parsed to 2021-08-31
    assert cleaned_df.loc[818, "DATE"] == "2021-08-31"

    # Check unparseable dates is 0
    assert report["date_validation"]["unparseable_dates_count"] == 0

    # Check excel serial dates converted is 1
    assert report["date_validation"]["excel_serials_converted_count"] == 1
    assert report["date_validation"]["excel_serials_converted"][0]["row_index"] == 818
    assert report["date_validation"]["excel_serials_converted"][0]["raw_value"] == 44439

    # Verify all DATE values are valid YYYY-MM-DD strings
    parsed_dates = pd.to_datetime(cleaned_df["DATE"], errors="coerce")
    assert parsed_dates.isna().sum() == 0, "Cleaned dataset contains invalid date values"


def test_no_duplicate_rows_or_region_date(preprocessing_results):
    """Verify 0 full duplicate rows and 0 duplicate Region+DATE pairs."""
    cleaned_df, report = preprocessing_results

    assert cleaned_df.duplicated().sum() == 0
    assert cleaned_df.duplicated(subset=["Region", "DATE"]).sum() == 0
    assert report["duplicates"]["exact_row_duplicates"] == 0
    assert report["duplicates"]["region_date_duplicates"] == 0


def test_no_missing_values(preprocessing_results):
    """Verify 0 missing values across all columns."""
    cleaned_df, report = preprocessing_results

    assert cleaned_df.isnull().sum().sum() == 0
    assert report["missing_values"]["total_missing_values"] == 0


def test_outlier_detection_preserves_rows(preprocessing_results):
    """Verify numerical outliers are reported and NOT deleted."""
    cleaned_df, report = preprocessing_results
    assert len(cleaned_df) == 1215

    outliers = report["numerical_outliers_iqr"]
    # Wind speed, rain, FFMC have known IQR outliers
    assert outliers["Ws"]["outlier_count"] > 0
    assert outliers["Rain"]["outlier_count"] > 0
    assert outliers["FFMC"]["outlier_count"] > 0

    # Confirm action is recorded as preserved
    for col, data in outliers.items():
        assert data["action"] == "reported_only_preserved_in_dataset"


def test_original_excel_preserved():
    """Verify original Excel file exists unmodified."""
    raw_df = load_raw_data(DEFAULT_INPUT_PATH)
    assert raw_df.shape == (1215, 13)
    assert "Bejaia 2021" in raw_df.columns
    assert "        Ws" in raw_df.columns
    assert "    Rain" in raw_df.columns


def test_output_artifacts_exist_and_valid():
    """Verify cleaned CSV and data quality report JSON exist on disk and are valid."""
    assert DEFAULT_OUTPUT_CSV.exists(), f"Missing cleaned CSV: {DEFAULT_OUTPUT_CSV}"
    assert DEFAULT_REPORT_JSON.exists(), f"Missing report JSON: {DEFAULT_REPORT_JSON}"

    # Read back CSV
    df = pd.read_csv(DEFAULT_OUTPUT_CSV)
    assert len(df) == 1215
    assert len(df.columns) == 13

    # Read back JSON
    with open(DEFAULT_REPORT_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert "dataset_overview" in report
    assert "descriptive_statistics" in report
    assert "class_distribution" in report
    assert "date_validation" in report
