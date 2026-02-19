"""Load and cache all Pittsfield Township assessment CSV data (2024-2026)."""

from pathlib import Path
import pandas as pd
import streamlit as st

DATA_ROOT = Path(__file__).parent.parent / "analysis"
YEARS = [2024, 2025, 2026]


@st.cache_data
def load_all_data() -> dict:
    """Load all CSVs into a unified dict. Cached by Streamlit."""
    return {
        "sales": _load_sales(),
        "ecf_analysis": _load_ecf_analysis(),
        "ecf_summaries": _load_ecf_summaries(),
        "land": _load_land_analysis(),
        "land_adj": _load_land_adjustments(),
        "all_areas": _get_all_areas(),
    }


def _load_sales() -> dict[int, pd.DataFrame]:
    """Load Sales Analysis CSVs. 2025 is missing some columns."""
    frames = {}
    for year in YEARS:
        path = DATA_ROOT / str(year) / f"{year}_Residential_Sales_Analysis.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        # Normalize ECF_Area: strip whitespace and leading apostrophes
        if "ECF_Area" in df.columns:
            df["ECF_Area"] = df["ECF_Area"].str.strip().str.lstrip("'")
        # Ensure numeric columns exist (2025 is missing some)
        for col in ["Sale_Price", "Adj_Sale", "Asd_When_Sold", "Asd_Adj_Sale", "Cur_Appraisal"]:
            if col not in df.columns:
                df[col] = pd.NA
            else:
                df[col] = pd.to_numeric(df[col].str.replace(r"[\$,]", "", regex=True), errors="coerce")
        df["_year"] = year
        frames[year] = df
    return frames


def _load_ecf_analysis() -> dict[int, pd.DataFrame]:
    """Load per-property ECF Analysis CSVs."""
    frames = {}
    for year in YEARS:
        path = DATA_ROOT / str(year) / f"{year}_Residential_ECF_Analysis.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        if "ECF_Area_Code" in df.columns:
            df["ECF_Area_Code"] = df["ECF_Area_Code"].str.strip().str.lstrip("'")
        # Parse numeric columns
        for col in ["Sale_Price", "Adj_Sale", "Land_Value", "Land_Yard", "Bldg_Residual", "Cost_Man"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].str.replace(r"[\$,]", "", regex=True), errors="coerce")
        if "ECF" in df.columns:
            df["ECF"] = pd.to_numeric(df["ECF"], errors="coerce")
            # Filter out obvious parsing errors (ECF should be 0.1-5.0)
            df.loc[~df["ECF"].between(0.1, 5.0), "ECF"] = pd.NA
        df["_year"] = year
        frames[year] = df
    return frames


def _load_ecf_summaries() -> dict[int, pd.DataFrame]:
    """Load ECF area-level summaries. Handle 2024's multi-row anomaly."""
    frames = {}
    for year in YEARS:
        path = DATA_ROOT / str(year) / f"{year}_Residential_ECF_Analysis_ECF_Summaries.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        df["ECF_Area"] = df["ECF_Area"].str.strip().str.lstrip("'")
        df["Ave_ECF"] = pd.to_numeric(df["Ave_ECF"], errors="coerce")
        # For areas with multiple rows (like AR-4 in 2024), take the FIRST value.
        # The 2024 file has individual property ECFs mixed in, but the first row
        # per area is the official area average (matches the existing analysis).
        df = df.groupby("ECF_Area", as_index=False).agg(
            Subdivision=("Subdivision", "first"),
            Ave_ECF=("Ave_ECF", "first"),
        )
        df["_year"] = year
        frames[year] = df
    return frames


def _load_land_analysis() -> dict[int, pd.DataFrame]:
    """Load Land Analysis CSVs. Normalize year-specific column names."""
    frames = {}
    for year in YEARS:
        path = DATA_ROOT / str(year) / f"{year}_Residential_Land_Analysis.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        # Normalize area code column
        for col in ["ECF_Area", "Area_Code"]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.lstrip("'")
        # Normalize year-specific land value columns
        prior_col = f"Land_Value_{year - 1}"
        current_col = f"Land_Value_{year}"
        if prior_col in df.columns:
            df = df.rename(columns={prior_col: "Land_Value_Prior"})
        if current_col in df.columns:
            df = df.rename(columns={current_col: "Land_Value_Current"})
        # Parse numeric columns
        for col in ["Sale_Price", "Adj_Sale", "Land_Residual", "Land_Value_Prior",
                     "Land_Value_Current", "Ratio_LV_SP", "Total_Acres"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].str.replace(r"[\$,]", "", regex=True), errors="coerce")
        df["_year"] = year
        frames[year] = df
    return frames


def _load_land_adjustments() -> dict[int, pd.DataFrame]:
    """Load Land Adjustment factor CSVs."""
    frames = {}
    for year in YEARS:
        path = DATA_ROOT / str(year) / f"{year}_Residential_Land_Analysis_Adjustments.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        if "Area_Code" in df.columns:
            df["Area_Code"] = df["Area_Code"].str.strip().str.lstrip("'")
        if "Adjust_Factor" in df.columns:
            df["Adjust_Factor"] = pd.to_numeric(df["Adjust_Factor"], errors="coerce")
        df["_year"] = year
        frames[year] = df
    return frames


def _get_all_areas() -> list[str]:
    """Get sorted list of all unique ECF area codes across all data."""
    areas = set()
    for year in YEARS:
        path = DATA_ROOT / str(year) / f"{year}_Residential_ECF_Analysis_ECF_Summaries.csv"
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            for code in df["ECF_Area"].str.strip().str.lstrip("'").dropna().unique():
                if len(code) <= 10 and code.replace("-", "").replace(".", "").isalnum():
                    areas.add(code)
        # Also check land adjustments for areas not in ECF summaries
        path = DATA_ROOT / str(year) / f"{year}_Residential_Land_Analysis_Adjustments.csv"
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            if "Area_Code" in df.columns:
                for code in df["Area_Code"].str.strip().str.lstrip("'").dropna().unique():
                    if len(code) <= 10 and code.replace("-", "").replace(".", "").isalnum():
                        areas.add(code)
    return sorted(areas)
