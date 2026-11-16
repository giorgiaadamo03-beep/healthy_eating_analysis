# Is Healthy Eating a Luxury?

### A global analysis of the cost and affordability of a healthy diet

This project explores whether a healthy diet is economically accessible across countries, how affordability has changed over time, and which socioeconomic factors are associated with these differences. It combines **FAOSTAT** data on the *Cost and Affordability of a Healthy Diet* with socioeconomic indicators from the **World Bank**.

The analysis covers **2017–2025**, subject to data availability, and combines exploratory data visualization with supervised and unsupervised machine learning.

## Explore the Interactive Dashboard

As part of the preliminary exploratory analysis, we developed an **interactive dashboard** to explore the data in greater detail, particularly at the **individual-country level**.

The dashboard provides interactive visualizations of geographic differences, trends over time, socioeconomic indicators, and food-group costs. Users can select and compare countries, continents, years, and indicators to investigate patterns beyond the main findings presented in the analysis.

**To explore the dashboard:** run the notebook and open the generated HTML file in `outputs/02_eda/dashboard/`. The dashboard is generated locally and is not included in the repository's tracked outputs.

## Research questions

1. **Where Is Healthy Eating a Luxury?** — How do the cost and affordability of a healthy diet differ across countries and continents?
2. **Does Wealth Guarantee Affordability?** — How are income and other economic indicators related to affordability?
3. **How Have Cost and Affordability Changed Over Time?** — What trends emerge between 2017 and 2025?
4. **What's Behind the Price of a Healthy Diet?** — How do different food groups contribute to its cost?
5. **Can We Identify Countries with Similar Socioeconomic Characteristics?** — What groups emerge from unsupervised learning, and how do their affordability outcomes differ?
6. **Can We Predict Affordability?** — How well can socioeconomic indicators predict the percentage of people unable to afford a healthy diet?


## Data sources

The project uses two publicly available sources. **The original data files are included in the project directory when distributed with the complete dataset**, but they can also be downloaded from the official sources below.

| Source | Data | Location in this project |
| --- | --- | --- |
| [FAOSTAT — Cost and Affordability of a Healthy Diet](https://www.fao.org/faostat/en/#data/CAHD) | Cost of a healthy diet (**CoHD**), percentage (**PUA**) and number (**NUA**) of people unable to afford it; food-group costs for 2021 | `data/FAOSTAT_data_en_11-3-2026.xlsx` |
| [World Bank — World Development Indicators](https://databank.worldbank.org/source/world-development-indicators) | GDP per capita (PPP), inflation, unemployment, urban population, population, surface area, Gini index, and poverty indicators | `data/world_bank/` |

### Downloading the original data

**FAOSTAT:** Open the *Cost and Affordability of a Healthy Diet* dataset, select the relevant indicators and years, and export the data in Excel format. The scripts currently expect the filename `FAOSTAT_data_en_11-3-2026.xlsx` inside `data/`. If the downloaded filename differs, rename it or update the corresponding file path in the code.

**World Bank:** Download the required indicators as CSV files from the World Development Indicators database and extract them into `data/world_bank/`. The project currently uses these indicator codes:

| Indicator code | Description |
| --- | --- |
| `NY.GDP.PCAP.PP.CD` | GDP per capita, PPP (current international $) |
| `FP.CPI.TOTL.ZG` | Inflation, consumer prices (annual %) |
| `SL.UEM.TOTL.ZS` | Unemployment, total (% of total labor force) |
| `SP.URB.TOTL.IN.ZS` | Urban population (% of total population) |
| `SP.POP.TOTL` | Population, total |
| `AG.SRF.TOTL.K2` | Surface area (sq. km) |
| `SI.POV.GINI` | Gini index |
| `SI.POV.DDAY` | Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population); check the source metadata if using a different World Bank export vintage |

> **Reproducibility note:** Public datasets may be revised over time. For exact reproduction of the reported results, use the original files distributed with the project. The notebook **does not download datasets automatically**: it reads the files already present in `data/`.

## Project structure

```text
is_healthy_eating_a_luxury/
├── .venv/                          # Local Python environment 
├── data/
│   ├── FAOSTAT_data_en_11-3-2026.xlsx
│   ├── world_bank/                # Original World Bank CSV files
│   └── processed/                 # Generated datasets (excluded from Git)
│       ├── faostat_clean.csv
│       ├── panel_data.csv
│       ├── food_categories_2021.csv
│       └── ml/                    # Datasets prepared for machine learning
├── outputs/                        # Generated outputs (excluded from Git)
│   ├── 02_eda/                    # EDA figures, maps, tables, and dashboard
│   ├── 03_preprocessing/          # Data quality and preprocessing reports
│   ├── 04_unsupervised/           # PCA and clustering results
│   └── 05_supervised/             # Model evaluation and interpretation
├── scripts/                        # Python modules
│   ├── 01_data_cleaning.py
│   ├── 02_data_integration.py
│   ├── 03_eda_dashboard.py
│   ├── 03_eda.py
│   ├── 04_preprocessing.py
│   ├── 05_unsupervised.py
│   └── 06_supervised.py
├── healthy_diet_analysis.ipynb    # Main executable notebook
├── healthy_eating_analysis.pdf     # Presentation
├── requirements.txt
└── README.md
```


### What each module does

| Module | Purpose |
| --- | --- |
| `01_data_cleaning.py` | Loads and cleans the original FAOSTAT dataset. |
| `02_data_integration.py` | Prepares FAOSTAT indicators, imports World Bank data, harmonizes country identifiers, and builds the country-year panel and 2021 food-group dataset. |
| `03_eda_dashboard.py` | Builds the interactive HTML dashboard with geographic, temporal, economic, and food-category views. |
| `03_eda.py` | Produces the exploratory figures and analyses addressing the first four research questions. |
| `04_preprocessing.py` | Checks data quality and prepares separate datasets for supervised and unsupervised learning. |
| `05_unsupervised.py` | Performs feature transformations, PCA, K-Means clustering, cluster profiling, and visualization. |
| `06_supervised.py` | Trains and compares predictive models and evaluates their performance and feature contributions. |

The `.py` files provide reusable functions; **the notebook is the main entry point** and calls those functions in sequence.

## Getting started

**Requirements:** Python 3.11 is recommended, along with the dependencies listed in `requirements.txt`. Some static exports of Plotly figures may also require Kaleido and a compatible Chrome installation.

1. Download or clone the project and open its root folder in VS Code or Jupyter.
2. Check that the original FAOSTAT Excel file and World Bank CSV files are present in the locations described above.
3. Create and activate a virtual environment, then install the dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # macOS / Linux
   python -m pip install -r requirements.txt
   ```

   On Windows, activate the environment with `.venv\Scripts\activate` instead.

4. Open `healthy_diet_analysis.ipynb`, select the project's Python environment as the kernel, and run **Run All**.
5. Explore the tables and charts in the notebook. Generated datasets and exported visualizations are saved under `data/processed/` and `outputs/`.

**Important:** Run the notebook from the **project root** and keep the original directory structure. Files in `data/processed/` are generated outputs; they are **not required as starting inputs** when running the full pipeline from the original data. Both `data/processed/` and `outputs/` are excluded from the Git repository via `.gitignore` and are generated locally when running the notebook.

## Analytical workflow

The notebook follows this sequence:

1. **Data cleaning** — inspect and clean FAOSTAT indicators.
2. **Data integration** — match countries and combine FAOSTAT and World Bank data.
3. **Interactive dashboard** — generate an HTML dashboard for exploring the integrated data.
4. **Exploratory data analysis** — investigate geographic patterns, economic relationships, time trends, and food-group costs.
5. **Preprocessing** — assess missing data and outliers, and create analysis-specific datasets.
6. **Unsupervised learning** — standardize socioeconomic variables, use PCA for exploration, select K using silhouette scores, and fit K-Means.
7. **Supervised learning** — compare a baseline, linear regression, and Random Forest using a time-based train/validation/test split.
8. **Conclusions** — discuss findings and limitations.

### Key methodological choices

- **Unsupervised learning:** The analysis focuses on 2021. The current clustering module uses log GDP per capita (PPP), urbanization, unemployment, and inflation. CoHD and PUA are retained for **cluster interpretation**, rather than used as clustering features. The number of clusters is selected from candidate values using the silhouette score.
- **Supervised learning:** The target is **PUA**. The models are evaluated using a chronological split: **2017–2022** for training, **2023** for validation, and **2024–2025** for testing, with **MAE, RMSE, and R²** as evaluation metrics.
- **Missing observations:** Country and year coverage varies across indicators. Analysis datasets may therefore contain fewer observations than the integrated panel.

## Outputs and reproducibility

The notebook displays intermediate results, statistical summaries, model evaluations, and figures. It also saves selected outputs in the project's `outputs/` directory, including the interactive dashboard under `outputs/02_eda/dashboard/`.

To reproduce the analysis, use the same original data files, Python dependencies, and code version. **The notebook is designed to run from the raw files**, but execution should be checked in the target environment, especially for interactive HTML and static image export.

## Limitations

This is an observational analysis. Relationships between socioeconomic indicators and healthy-diet affordability should **not be interpreted as causal effects**. Missing data, differences in country coverage, and revisions to the original datasets can affect results. Model performance reflects the selected time split and available indicators.

