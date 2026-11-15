from pathlib import Path
import numpy as np
import pandas as pd


def find_project_root() -> Path:
    """Locate the project root from the script directory."""
    module_dir = Path(__file__).resolve().parent
    return module_dir.parent


PROJECT_ROOT = find_project_root()
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
ML_DIR = PROCESSED_DIR / 'ml'
REPORT_DIR = PROJECT_ROOT / 'outputs' / '03_preprocessing' / 'tables'

CLUSTER_YEAR = 2021
ECONOMIC_FEATURES = ['CoHD', 'GDP_PPP', 'Urban', 'Unemployment', 'Inflation']
SUPERVISED_FEATURES = ECONOMIC_FEATURES + ['Population_Density']
METADATA_COLUMNS = ['ISO3', 'Country', 'Continent']
CLUSTER_FEATURES = ['CoHD', 'log_GDP_PPP', 'Urban', 'Unemployment', 'Inflation']


def load_dataset(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a CSV and validate the columns required by downstream analyses."""
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}. Run 02_data_integration.py first.')
    data = pd.read_csv(path, encoding='utf-8-sig')
    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError(f'Missing columns in {path.name}: {missing}')
    return data


def clean_numeric_values(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert numeric fields and mark invalid values as missing."""
    data = data.copy()
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)

    for column in ['CoHD', 'GDP_PPP', 'Population_Density']:
        if column in data.columns:
            data.loc[data[column] <= 0, column] = np.nan
    for column in ['PUA', 'Urban']:
        if column in data.columns:
            data.loc[data[column].notna() & ~data[column].between(0, 100), column] = np.nan
    return data


def create_missing_value_report(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Summarize missing observations by variable."""
    return pd.DataFrame({
        'variable': columns,
        'missing_n': [int(data[column].isna().sum()) for column in columns],
        'missing_pct': [round(data[column].isna().mean() * 100, 2) for column in columns],
    })


def create_outlier_report(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Flag IQR-based outliers for diagnostics without removing observations."""
    rows = []
    for column in columns:
        values = data[column].dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            'variable': column, 'n': len(values), 'q1': q1, 'q3': q3,
            'lower_iqr': lower, 'upper_iqr': upper,
            'outliers_iqr': int(((values < lower) | (values > upper)).sum()),
        })
    return pd.DataFrame(rows, columns=[
        'variable', 'n', 'q1', 'q3', 'lower_iqr', 'upper_iqr', 'outliers_iqr'
    ])


def prepare_panel(data: pd.DataFrame) -> pd.DataFrame:
    """Validate unique country-year records and clean numeric variables."""
    panel = data.copy()
    panel['Year'] = pd.to_numeric(panel['Year'], errors='raise').astype(int)
    if panel.duplicated(['ISO3', 'Year']).any():
        raise ValueError('Duplicate ISO3-Year records found in the integrated panel.')
    return clean_numeric_values(panel, ['PUA'] + SUPERVISED_FEATURES)


def prepare_unsupervised_data(panel: pd.DataFrame, year: int = CLUSTER_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare all country records and complete cases for clustering.

    PUA is retained for post-clustering interpretation, not as a feature.
    """
    all_countries = panel.loc[
        panel['Year'].eq(year), METADATA_COLUMNS + ['Year'] + ECONOMIC_FEATURES + ['PUA']
    ].copy()
    all_countries['log_GDP_PPP'] = np.log(all_countries['GDP_PPP'])
    all_countries['complete_features'] = all_countries[CLUSTER_FEATURES].notna().all(axis=1)
    complete_cases = all_countries.loc[all_countries['complete_features']].drop(
        columns='complete_features'
    )
    return all_countries, complete_cases


def prepare_supervised_data(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep observed targets without imputing or scaling predictors.

    Preprocessing must be fitted on training data after the temporal split.
    """
    supervised = panel[METADATA_COLUMNS + ['Year', 'PUA'] + SUPERVISED_FEATURES].copy()
    supervised['log_GDP_PPP'] = np.log(supervised['GDP_PPP'])
    supervised = supervised.loc[supervised['PUA'].notna()].copy()
    return supervised.sort_values(['Year', 'ISO3']).reset_index(drop=True)


def create_dataset_summary(panel: pd.DataFrame, unsupervised: pd.DataFrame,
                           supervised: pd.DataFrame) -> pd.DataFrame:
    """Count rows and distinct countries in the prepared datasets."""
    return pd.DataFrame([
        {'dataset': 'panel_original', 'rows': len(panel), 'countries': panel.ISO3.nunique()},
        {'dataset': 'unsupervised_2021', 'rows': len(unsupervised), 'countries': unsupervised.ISO3.nunique()},
        {'dataset': 'supervised_panel', 'rows': len(supervised), 'countries': supervised.ISO3.nunique()},
    ])


def save_preprocessing_results(panel: pd.DataFrame, unsupervised_all: pd.DataFrame,
                               unsupervised: pd.DataFrame, supervised: pd.DataFrame,
                               ml_dir: Path = ML_DIR, report_dir: Path = REPORT_DIR) -> pd.DataFrame:
    """Write prepared datasets and quality reports to consistent directories."""
    ml_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    unsupervised_all.to_csv(ml_dir / 'unsupervised_2021_all.csv', index=False, encoding='utf-8-sig')
    unsupervised.to_csv(ml_dir / 'unsupervised_2021.csv', index=False, encoding='utf-8-sig')
    supervised.to_csv(ml_dir / 'supervised_panel.csv', index=False, encoding='utf-8-sig')

    create_missing_value_report(panel, ['PUA'] + SUPERVISED_FEATURES).to_csv(
        report_dir / 'missing_panel.csv', index=False)
    panel.groupby('Year')[['PUA'] + SUPERVISED_FEATURES].count().to_csv(
        report_dir / 'coverage_by_year.csv')
    create_outlier_report(panel, ['PUA'] + SUPERVISED_FEATURES).to_csv(
        report_dir / 'outliers_panel.csv', index=False)
    create_missing_value_report(unsupervised_all, CLUSTER_FEATURES + ['PUA']).to_csv(
        report_dir / 'missing_unsupervised_2021.csv', index=False)
    create_outlier_report(unsupervised, CLUSTER_FEATURES).to_csv(
        report_dir / 'outliers_unsupervised_2021.csv', index=False)

    summary = create_dataset_summary(panel, unsupervised, supervised)
    summary.to_csv(report_dir / 'dataset_summary.csv', index=False)
    return summary


def run_preprocessing(panel_data: pd.DataFrame | None = None,
                      panel_path: Path | None = None,
                      ml_dir: Path = ML_DIR,
                      report_dir: Path = REPORT_DIR,
                      save_outputs: bool = True) -> dict[str, pd.DataFrame]:
    """Prepare datasets from an in-memory panel or an integrated CSV.

    Supply panel_data to run the full workflow inside a notebook without
    depending on previously exported intermediate files.
    """
    if panel_data is not None:
        raw_panel = panel_data.copy()
        required = METADATA_COLUMNS + ['Year', 'PUA'] + SUPERVISED_FEATURES
        missing = sorted(set(required) - set(raw_panel.columns))
        if missing:
            raise ValueError(f'Missing columns in panel_data: {missing}')
    else:
        input_path = Path(panel_path) if panel_path is not None else PROCESSED_DIR / 'panel_data.csv'
        raw_panel = load_dataset(input_path, METADATA_COLUMNS + ['Year', 'PUA'] + SUPERVISED_FEATURES)

    panel = prepare_panel(raw_panel)
    unsupervised_all, unsupervised = prepare_unsupervised_data(panel)
    supervised = prepare_supervised_data(panel)
    if save_outputs:
        summary = save_preprocessing_results(
            panel, unsupervised_all, unsupervised, supervised, Path(ml_dir), Path(report_dir)
        )
    else:
        summary = create_dataset_summary(panel, unsupervised, supervised)
    return {
        'panel': panel,
        'unsupervised_all': unsupervised_all,
        'unsupervised': unsupervised,
        'supervised': supervised,
        'summary': summary,
    }
