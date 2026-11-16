from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "data" / "FAOSTAT_data_en_11-3-2026.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "faostat_clean.csv"

REQUIRED_COLUMNS = ["Area Code (M49)", "Area", "Item", "Year", "Unit", "Value"]
COLUMN_RENAMES = {"Area Code (M49)": "M49", "Area": "Country", "Item": "Indicator"}
PPP_UNIT = "Int$ (PPP) per person per day"


def load_faostat_data(input_path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Read the original FAOSTAT Excel file and normalize column names."""
    frame = pd.read_excel(input_path)
    frame.columns = frame.columns.str.strip()
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing FAOSTAT columns: {missing}")
    return frame


def clean_faostat_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the original country, unit, indicator, and type filters."""
    df = frame[REQUIRED_COLUMNS].rename(columns=COLUMN_RENAMES).copy()

    # Exclude the aggregated China area while retaining mainland China.
    df = df.loc[df["Country"].ne("China")].copy()

    # Keep PPP-denominated cost indicators without dropping PUA or NUA.
    is_cost = df["Indicator"].str.startswith("Cost of", na=False)
    df = df.loc[~is_cost | df["Unit"].eq(PPP_UNIT)].copy()

    # Exclude alternative averages that differ from standard indicators.
    df = df.loc[
        ~df["Indicator"].str.contains(
            r", (?:simple|weighted) average$", regex=True, na=False
        )
    ].copy()

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["M49"] = pd.to_numeric(df["M49"], errors="coerce").astype("Int64")

    return df.sort_values(
        ["Country", "Indicator", "Year", "Unit"]
    ).reset_index(drop=True)


def build_indicator_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Count records and available values for each indicator and unit."""
    return (
        df.groupby(["Indicator", "Unit"], dropna=False)["Value"]
        .agg(rows="size", available="count")
        .reset_index()
    )


def summarize_faostat_data(df: pd.DataFrame) -> dict:
    """Return a compact summary for logging or notebook reporting."""
    return {
        "rows": len(df),
        "countries_or_areas": df["Country"].nunique(),
        "first_year": df["Year"].min(),
        "last_year": df["Year"].max(),
        "indicators": df["Indicator"].nunique(),
        "missing_values": int(df["Value"].isna().sum()),
        "available_m49": int(df["M49"].notna().sum()),
    }


def save_cleaned_data(df: pd.DataFrame, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Save the cleaned dataset to the shared processed-data directory."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
