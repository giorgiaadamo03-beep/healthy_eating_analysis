from pathlib import Path
import re
import pandas as pd
import pycountry
import pycountry_convert as pc


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WORLD_BANK_DIR = PROJECT_ROOT / "data" / "world_bank"
YEARS = list(range(2017, 2026))
PPP_UNIT = "Int$ (PPP) per person per day"

WORLD_BANK_INDICATORS = {
    "NY.GDP.PCAP.PP.CD": "GDP_PPP",
    "SI.POV.GINI": "Gini",
    "SI.POV.DDAY": "Poverty",
    "FP.CPI.TOTL.ZG": "Inflation",
    "SL.UEM.TOTL.ZS": "Unemployment",
    "SP.URB.TOTL.IN.ZS": "Urban",
    "SP.POP.TOTL": "Population",
    "AG.SRF.TOTL.K2": "Surface_km2",
}
FOOD_INDICATORS = {
    "Cost of animal source foods": "Animal_Foods",
    "Cost of fruits": "Fruits",
    "Cost of legumes, nuts and seeds": "Legumes_Nuts_Seeds",
    "Cost of oils and fats": "Oils_Fats",
    "Cost of starchy staples": "Starchy_Staples",
    "Cost of vegetables": "Vegetables",
}
LONGITUDINAL_INDICATORS = {
    "Cost of a healthy diet (CoHD)": "CoHD",
    "Prevalence of unaffordability (PUA)": "PUA",
    "Number of people unable to afford a healthy diet (NUA)": "NUA",
}
COUNTRY_CODES = {
    "Bahamas": "BHS", "Bolivia (Plurinational State of)": "BOL",
    "China, Hong Kong SAR": "HKG", "China, mainland": "CHN",
    "Congo": "COG", "Curaçao": "CUW", "Côte d'Ivoire": "CIV",
    "Democratic Republic of the Congo": "COD", "Egypt": "EGY",
    "Gambia": "GMB", "Iran (Islamic Republic of)": "IRN",
    "Kyrgyzstan": "KGZ", "Lao People's Democratic Republic": "LAO",
    "Netherlands (Kingdom of the)": "NLD", "Palestine": "PSE",
    "Republic of Korea": "KOR", "Republic of Moldova": "MDA",
    "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT", "Slovakia": "SVK",
    "Somalia": "SOM", "Türkiye": "TUR",
    "United Kingdom of Great Britain and Northern Ireland": "GBR",
    "United Republic of Tanzania": "TZA", "United States of America": "USA",
}
CONTINENT_NAMES = {
    "AF": "Africa", "AS": "Asia", "EU": "Europe",
    "NA": "North America", "SA": "South America",
    "OC": "Oceania", "AN": "Antarctica",
}


def load_cleaned_faostat(path: Path = PROCESSED_DIR / "faostat_clean.csv") -> pd.DataFrame:
    """Load cleaned FAOSTAT observations for the analysis period."""
    frame = pd.read_csv(path)
    return frame.loc[frame["Year"].isin(YEARS)].copy()


def prepare_food_categories(faostat: pd.DataFrame) -> pd.DataFrame:
    """Pivot the six 2021 PPP food-cost indicators into country-level columns."""
    food = faostat.loc[
        faostat["Year"].eq(2021)
        & faostat["Indicator"].isin(FOOD_INDICATORS)
        & faostat["Unit"].eq(PPP_UNIT),
        ["Country", "Indicator", "Value"],
    ].copy()
    if food.duplicated(["Country", "Indicator"]).any():
        raise ValueError("Duplicate 2021 food-category records by country and indicator.")
    food["Indicator"] = food["Indicator"].map(FOOD_INDICATORS)
    result = food.pivot(index="Country", columns="Indicator", values="Value").reset_index()
    result.columns.name = None
    return result


def prepare_longitudinal_faostat(faostat: pd.DataFrame) -> pd.DataFrame:
    """Pivot CoHD, PUA, and NUA into one row per country and year."""
    frame = faostat.loc[faostat["Indicator"].isin(LONGITUDINAL_INDICATORS)].copy()
    if frame.duplicated(["Country", "Year", "Indicator"]).any():
        raise ValueError("Duplicate FAOSTAT records by country, year, and indicator.")
    frame["Indicator"] = frame["Indicator"].map(LONGITUDINAL_INDICATORS)
    result = frame.pivot(
        index=["Country", "Year"], columns="Indicator", values="Value"
    ).reset_index()
    result.columns.name = None
    return result


def read_world_bank_indicator(path: Path, variable: str) -> pd.DataFrame:
    """Read a World Bank CSV and convert year columns into long format."""
    frame = pd.read_csv(path, skiprows=4, encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    needed = ["Country Name", "Country Code"] + [str(year) for year in YEARS]
    missing = set(needed) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    frame = frame[needed].melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=[str(year) for year in YEARS],
        var_name="Year", value_name=variable,
    )
    frame["Year"] = frame["Year"].astype(int)
    frame[variable] = pd.to_numeric(frame[variable], errors="coerce")
    return frame


def load_world_bank_data(directory: Path = WORLD_BANK_DIR) -> tuple[pd.DataFrame, int]:
    """Merge World Bank indicators and exclude aggregate economy codes."""
    combined = None
    for code, variable in WORLD_BANK_INDICATORS.items():
        files = list(directory.glob(f"API_{code}_*.csv"))
        if len(files) != 1:
            raise FileNotFoundError(
                f"Expected one CSV for {code}; found {len(files)}."
            )
        part = read_world_bank_indicator(files[0], variable)
        combined = part if combined is None else combined.merge(
            part.drop(columns="Country Name"),
            on=["Country Code", "Year"], how="outer", validate="one_to_one",
        )
    valid_codes = {country.alpha_3 for country in pycountry.countries}
    valid_codes.add("XKX")
    initial_count = combined["Country Code"].nunique()
    combined = combined.loc[combined["Country Code"].isin(valid_codes)].copy()
    excluded_count = initial_count - combined["Country Code"].nunique()
    return combined, excluded_count


def normalize_country_name(name: str) -> str:
    """Normalize country names for exact matching after punctuation removal."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def match_faostat_countries(
    faostat: pd.DataFrame, world_bank: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Match countries automatically, then apply verified ISO3 overrides."""
    wb_names = world_bank[["Country Name", "Country Code"]].drop_duplicates().copy()
    wb_names["key"] = wb_names["Country Name"].map(normalize_country_name)
    ambiguous = wb_names["key"].duplicated(keep=False)
    ambiguous_names = wb_names.loc[
        ambiguous, ["Country Name", "Country Code"]
    ].copy()
    wb_names = wb_names.loc[~ambiguous]

    source = faostat.copy()
    source["key"] = source["Country"].map(normalize_country_name)
    matched = source.merge(wb_names, on="key", how="left", validate="many_to_one")
    auto_matched = matched.loc[
        matched["Country Code"].notna(), "Country"
    ].nunique()
    auto_unmatched = matched.loc[
        matched["Country Code"].isna(), "Country"
    ].drop_duplicates().sort_values()

    manual = matched["Country"].map(COUNTRY_CODES)
    conflict = (
        matched["Country Code"].notna()
        & manual.notna()
        & matched["Country Code"].ne(manual)
    )
    if conflict.any():
        names = ", ".join(matched.loc[conflict, "Country"].unique())
        raise ValueError(f"Conflicting country mappings: {names}")
    matched["Country Code"] = matched["Country Code"].fillna(manual)
    matched.loc[
        ~matched["Country Code"].isin(world_bank["Country Code"]),
        "Country Code",
    ] = pd.NA
    unmatched = matched.loc[
        matched["Country Code"].isna(), "Country"
    ].drop_duplicates().sort_values()
    report = {
        "ambiguous_world_bank_names": ambiguous_names,
        "automatically_matched_countries": auto_matched,
        "automatically_unmatched_countries": auto_unmatched,
        "finally_matched_countries": matched.loc[
            matched["Country Code"].notna(), "Country"
        ].nunique(),
        "unmatched_countries": unmatched,
    }
    return matched.loc[matched["Country Code"].notna()].copy(), report


def integrate_country_year_data(
    matched: pd.DataFrame, world_bank: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Join country-year observations and derive population density."""
    combined = matched.merge(
        world_bank.drop(columns="Country Name"),
        on=["Country Code", "Year"], how="left", validate="many_to_one",
    )
    # Carry 2024 surface area forward only when the 2025 value is missing.
    surface_2024 = (
        combined.loc[combined["Year"].eq(2024), ["Country Code", "Surface_km2"]]
        .drop_duplicates(subset=["Country Code"])
        .set_index("Country Code")["Surface_km2"]
    )
    missing_2025 = combined["Year"].eq(2025) & combined["Surface_km2"].isna()
    combined.loc[missing_2025, "Surface_km2"] = (
        combined.loc[missing_2025, "Country Code"].map(surface_2024).to_numpy()
    )
    filled_count = int(combined.loc[missing_2025, "Surface_km2"].notna().sum())
    combined["Population_Density"] = (
        combined["Population"] / combined["Surface_km2"].where(
            combined["Surface_km2"] > 0
        )
    )
    return combined, filled_count


def iso3_to_continent(iso3: str) -> str:
    """Map an ISO3 country code to its continent, including Kosovo."""
    if iso3 == "XKX":
        return "Europe"
    try:
        iso2 = pycountry.countries.get(alpha_3=str(iso3)).alpha_2
        continent_code = pc.country_alpha2_to_continent_code(iso2)
        return CONTINENT_NAMES.get(continent_code, "Unclassified")
    except (AttributeError, KeyError, TypeError, ValueError):
        return "Unclassified"


def build_panel(combined: pd.DataFrame) -> pd.DataFrame:
    """Build the country-year panel with continent labels and unique ISO3-year keys."""
    columns = (
        ["Country Code", "Country", "Year", "CoHD", "PUA", "NUA"]
        + list(WORLD_BANK_INDICATORS.values())
        + ["Population_Density"]
    )
    panel = combined[columns].rename(columns={"Country Code": "ISO3"}).copy()
    continent_map = {
        code: iso3_to_continent(code) for code in panel["ISO3"].dropna().unique()
    }
    panel["Continent"] = panel["ISO3"].map(continent_map).fillna("Unclassified")
    if panel.duplicated(["ISO3", "Year"]).any():
        duplicates = panel.loc[
            panel.duplicated(["ISO3", "Year"], keep=False),
            ["ISO3", "Country", "Year"],
        ].sort_values(["ISO3", "Year", "Country"])
        raise ValueError(
            "Duplicate ISO3-year panel keys:\n" + duplicates.to_string(index=False)
        )
    return panel


def build_food_dataset(panel: pd.DataFrame, food: pd.DataFrame) -> pd.DataFrame:
    """Combine 2021 panel data with the six food-cost categories."""
    food_2021 = panel.loc[panel["Year"].eq(2021)].drop(columns="Year").merge(
        food, on="Country", how="left", validate="one_to_one"
    )
    columns = (
        ["ISO3", "Country", "Continent"]
        + list(FOOD_INDICATORS.values())
        + ["CoHD", "PUA", "NUA"]
        + list(WORLD_BANK_INDICATORS.values())
        + ["Population_Density"]
    )
    return food_2021[columns]


def save_integrated_datasets(
    panel: pd.DataFrame,
    food_2021: pd.DataFrame,
    directory: Path = PROCESSED_DIR,
) -> dict[str, Path]:
    """Save both datasets under stable filenames used by downstream scripts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "panel": directory / "panel_data.csv",
        "food_2021": directory / "food_categories_2021.csv",
    }
    panel.to_csv(paths["panel"], index=False, encoding="utf-8-sig")
    food_2021.to_csv(paths["food_2021"], index=False, encoding="utf-8-sig")
    return paths
