from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go

CREAM = "#FBFBEC"
DARK_GREEN = "#07391E"
GREEN = "#4F7E52"
ORANGE = "#E5871E"
PURPLE = "#8F6688"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_dashboard(panel_path=None, food_path=None, output_dir=None, *, panel_data=None, food_data=None):
    """Build the complete interactive HTML dashboard and return its path.

    Args:
        panel_path: Optional path to the integrated country-year panel.
        food_path: Optional path to the 2021 food categories dataset.
        output_dir: Optional directory for the generated HTML dashboard.
        panel_data: Optional in-memory panel DataFrame from the notebook.
        food_data: Optional in-memory food categories DataFrame from the notebook.
        DataFrames take precedence over CSV paths when provided.
    """
    BASE_DIR = PROJECT_ROOT
    DATA_PATH = Path(panel_path) if panel_path is not None else BASE_DIR / "data" / "processed" / "panel_data.csv"
    OUTPUT_DIR = Path(output_dir) if output_dir is not None else BASE_DIR / "outputs" / "02_eda" / "dashboard"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    if panel_data is not None:
        df = panel_data.copy()
    else:
        if not DATA_PATH.is_file():
            raise FileNotFoundError(f"Integrated panel not found: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    required = {"ISO3", "Country", "Year", "CoHD", "PUA", "Continent", "Population"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    for column in ("CoHD", "PUA", "Population"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    print(f"Observations: {len(df)}")
    print(f"Countries: {df['ISO3'].nunique()}")
    print(f"Years: {int(df['Year'].min())}–{int(df['Year'].max())}")
    print("\nData availability by year:")
    print(df.groupby("Year")[["CoHD", "PUA"]].count().to_string())
    print("\nDescriptive statistics:")
    print(df[["CoHD", "PUA"]].describe().round(2).to_string())

    # Use a fixed color scale across years for each indicator.
    indicators = {
        "CoHD": {"label": "Daily cost of a healthy diet (PPP $)", "colorscale": [[0, CREAM], [0.5, ORANGE], [1, DARK_GREEN]]},
        "PUA": {"label": "Population unable to afford a healthy diet (%)", "colorscale": [[0, CREAM], [0.5, ORANGE], [1, DARK_GREEN]]},
    }
    CONTINENT_COLORS = {
        "Africa": "#E5871E", 
        "Asia": "#8F6688", 
        "Europe": "#187D9B",
        "North America": "#B83E65", 
        "South America": "#659B32",
        "Oceania": "#A66B19"
    }
    continents = sorted(df["Continent"].dropna().unique())
    continent_by_iso = dict(df[["ISO3", "Continent"]].dropna().drop_duplicates("ISO3").itertuples(index=False, name=None))
    continent_options = "".join(f'<option value="{c}">{c}</option>' for c in continents)
    years = sorted(int(year) for year in df["Year"].dropna().unique())
    fig = go.Figure()
    trace_keys = []

    for indicator, config in indicators.items():
        maximum = df[indicator].max(skipna=True)
        for year in years:
            subset = df.loc[df["Year"] == year, ["ISO3", "Country", indicator]].dropna()
            fig.add_trace(
                go.Choropleth(
                    locations=subset["ISO3"],
                    z=subset[indicator],
                    text=subset["Country"],
                    locationmode="ISO-3",
                    colorscale=config["colorscale"],
                    zmin=0,
                    zmax=maximum,
                    colorbar=dict(
                        title=None,
                        x=1.01,
                        xanchor="left",
                        y=0.5,
                        len=0.65,
                        thickness=15,
                        outlinewidth=0,
                        tickfont=dict(size=12, color=DARK_GREEN),
                    ),
                    hovertemplate="<b>%{text}</b><br>Value: %{z:.2f}<extra></extra>",
                    visible=False,
                )
            )
            trace_keys.append((indicator, year))

    initial_key = ("PUA", years[-1])
    fig.data[trace_keys.index(initial_key)].visible = True


    fig.update_layout(
        title=dict(text=f"<b>{indicators[initial_key[0]]['label']} — {initial_key[1]}</b>"),
        template="plotly_white",
        font=dict(family="Arial, Helvetica, sans-serif", color=DARK_GREEN),
        title_font=dict(family="Arial, Helvetica, sans-serif", size=24, color=DARK_GREEN),
        title_x=0.045,
        title_y=0.96,
        paper_bgcolor=CREAM,
        plot_bgcolor=CREAM,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            projection_rotation=dict(lon=15),
            bgcolor=CREAM, showland=True, landcolor=CREAM,
            showcountries=True, countrycolor=GREEN, coastlinecolor=GREEN,
        ),
        margin=dict(l=8, r=50, t=75, b=10),
        height=690,
    )

    fig.update_layout(font=dict(family="Arial, Helvetica, sans-serif", color=DARK_GREEN, size=14), paper_bgcolor=CREAM, plot_bgcolor=CREAM)
    # Calculate population-weighted averages separately for each year.
    # Exclude countries with missing or non-positive population.
    def weighted_mean(group, metric):
        valid = group.loc[group[metric].notna() & group["Population"].notna() & (group["Population"] > 0)]
        if valid.empty:
            return float("nan")
        return (valid[metric] * valid["Population"]).sum() / valid["Population"].sum()

    trend_fig = go.Figure()
    trend_keys = []
    trend_axis_ranges = {
        "PUA": [0, max(100, float(df["PUA"].max(skipna=True)) * 1.05)],
        "CoHD": [0, float(df["CoHD"].max(skipna=True)) * 1.05],
    }
    for metric in ("CoHD", "PUA"):
        valid = df.dropna(subset=[metric])
        complete = valid.groupby("ISO3")["Year"].nunique()
        complete_iso = set(complete[complete == len(years)].index)
        balanced = valid[valid["ISO3"].isin(complete_iso)]
        mean_series = balanced.groupby("Year").apply(lambda g: weighted_mean(g, metric), include_groups=False).reindex(years)
        trend_fig.add_trace(go.Scatter(
            x=years, y=mean_series, mode="lines+markers",
            name="World average (population-weighted)",
            line=dict(color=DARK_GREEN, width=4), marker=dict(size=9),
            visible=(metric == "PUA"),
            hovertemplate="Year: %{x}<br>Average: %{y:.2f}<extra></extra>",
        ))
        trend_keys.append([metric, "__average__"])
        for continent in continents:
            regional = balanced.loc[balanced["Continent"] == continent]
            regional_mean = regional.groupby("Year").apply(lambda g: weighted_mean(g, metric), include_groups=False).reindex(years)
            color = CONTINENT_COLORS.get(continent, PURPLE)
            trend_fig.add_trace(go.Scatter(
                x=years, y=regional_mean, mode="lines+markers",
                name=f"Average {continent}", line=dict(color=color, width=3, dash="dash"),
                marker=dict(color=color, size=9), visible=False,
                hovertemplate=f"Average {continent}<br>Year: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
            ))
            trend_keys.append([metric, f"__continent__:{continent}"])

        for iso3, country in sorted(df[["ISO3", "Country"]].dropna().drop_duplicates().itertuples(index=False, name=None), key=lambda item: item[1]):
            values = df[df["ISO3"] == iso3].set_index("Year")[metric].reindex(years)
            color = CONTINENT_COLORS.get(continent_by_iso.get(iso3), PURPLE)
            trend_fig.add_trace(go.Scatter(
                x=years, y=values, mode="lines+markers", name=country,
                line=dict(color=color, width=2.5), marker=dict(color=color, size=6),
                connectgaps=False, visible=False,
                hovertemplate=f"{country}<br>Year: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
            ))
            trend_keys.append([metric, iso3])
        print(f"Countries with complete series {metric}: {len(complete_iso)}")

    trend_fig.update_layout(
        title=dict(text="<b>Healthy diet affordability over time · 2017–2025</b>"),
        template="plotly_white", height=620,
        font=dict(family="Arial, Helvetica, sans-serif", color=DARK_GREEN),
        title_font=dict(size=22, color=DARK_GREEN),
        margin=dict(l=70, r=35, t=90, b=80),
        xaxis=dict(title="Year", tickmode="array", tickvals=years),
        yaxis=dict(title="Population unable to afford a healthy diet (%)", range=trend_axis_ranges["PUA"], autorange=False),
        hovermode="closest", showlegend=False, legend=dict(orientation="h", y=-0.2),
    )

    trend_fig.update_layout(font=dict(family="Arial, Helvetica, sans-serif", color=DARK_GREEN, size=14), paper_bgcolor=CREAM, plot_bgcolor=CREAM)
    trend_fig.update_xaxes(gridcolor="#D7DFD2", linecolor=GREEN, tickfont=dict(color=DARK_GREEN))
    trend_fig.update_yaxes(gridcolor="#D7DFD2", linecolor=GREEN, tickfont=dict(color=DARK_GREEN))


    FOOD_PATH = Path(food_path) if food_path is not None else BASE_DIR / "data" / "processed" / "food_categories_2021.csv"
    if food_data is not None:
        food_df = food_data.copy()
    else:
        if not FOOD_PATH.is_file():
            raise FileNotFoundError(f"Food categories dataset not found: {FOOD_PATH}")
        food_df = pd.read_csv(FOOD_PATH)
    # Display labels for the six FAOSTAT food cost categories.
    food_categories = {
        "Animal_Foods": "Animal-source foods",
        "Fruits": "Fruits",
        "Legumes_Nuts_Seeds": "Legumes, nuts and seeds",
        "Oils_Fats": "Oils and fats",
        "Starchy_Staples": "Starchy staples",
        "Vegetables": "Vegetables",
    }

    # The food categories dataset may not contain population. Reuse the
    # 2021 population from the already integrated panel when necessary.
    if "Population" not in food_df.columns:
        population_2021 = (
            df.loc[df["Year"] == 2021, ["ISO3", "Population"]]
            .drop_duplicates(subset="ISO3")
        )
        food_df = food_df.merge(population_2021, on="ISO3", how="left", validate="many_to_one")

    food_required = {"ISO3", "Country", "Population", *food_categories}
    if food_required - set(food_df.columns):
        raise ValueError(f"Missing columns in food dataset: {sorted(food_required - set(food_df.columns))}")
    for col in (*food_categories, "Population"):
        food_df[col] = pd.to_numeric(food_df[col], errors="coerce")
    food_df["Continent"] = food_df["ISO3"].map(continent_by_iso).fillna("Unclassified")
    food_records = []
    for row in food_df.to_dict("records"):
        food_records.append({
            "iso3": str(row["ISO3"]), "country": str(row["Country"]),
            "continent": str(row["Continent"]),
            "population": None if pd.isna(row["Population"]) or row["Population"] <= 0 else float(row["Population"]),
            "categories": {key: None if pd.isna(row[key]) else float(row[key]) for key in food_categories},
            "cohd": None if "CoHD" not in row or pd.isna(row["CoHD"]) else float(row["CoHD"]),
        })
    # Build the parallel-coordinates chart for the six food groups (2021).
    food_parallel_fig = go.Figure()
    food_parallel_fig.update_layout(
        template="plotly_white", height=630,
        margin=dict(l=80, r=65, t=100, b=85),
        title="<b>Food cost profiles · Parallel Coordinates (2021)</b>",
        font=dict(family="Arial, Helvetica, sans-serif", color=DARK_GREEN),
    )
    food_parallel_fig.update_layout(font=dict(family="Arial, Helvetica, sans-serif", color=DARK_GREEN, size=14), paper_bgcolor=CREAM, plot_bgcolor=CREAM)
    food_parallel_html = food_parallel_fig.to_html(
        include_plotlyjs=False, full_html=False, div_id="food-parallel-chart"
    )

    output_path = OUTPUT_DIR / "healthy_diet_dashboard.html"
    plot_html = fig.to_html(include_plotlyjs=True, full_html=False, div_id="world-map")

    trend_html = trend_fig.to_html(include_plotlyjs=False, full_html=False, div_id="trend-chart")


    economic_vars = {
        "GDP_PPP": "GDP per capita (PPP $)",
        "Inflation": "Inflation (%)",
        "Unemployment": "Unemployment (%)",
        "Urban": "Urban population (%)",
    }
    for col in economic_vars:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Keep x-axis limits fixed across years for each metric.
    scatter_x_ranges = {}
    for variable in economic_vars:
        values = df.loc[df["PUA"].notna(), variable].dropna()
        if values.empty:
            scatter_x_ranges[variable] = [0, 1]
        else:
            lower, upper = float(values.min()), float(values.max())
            padding = (upper - lower) * 0.05 if upper > lower else max(abs(lower) * 0.05, 1.0)
            scatter_x_ranges[variable] = [lower - padding, upper + padding]

    scatter_data = {}
    for variable in economic_vars:
        scatter_data[variable] = {}
        for year in years:
            subset = df.loc[df["Year"] == year, ["ISO3", "Country", "Continent", "PUA", variable]].dropna()
            scatter_data[variable][year] = [
                {"iso3": str(r.ISO3), "country": str(r.Country), "continent": str(r.Continent), "x": float(getattr(r, variable)), "y": float(r.PUA)}
                for r in subset.itertuples(index=False)
            ]
    scatter_fig = go.Figure(go.Scatter(
        x=[], y=[], mode="markers", marker=dict(size=10, opacity=0.7, color=PURPLE),
        customdata=[], hovertemplate="<b>%{customdata}</b><br>X: %{x:,.2f}<br>PUA: %{y:.2f}%<extra></extra>"
    ))
    scatter_fig.update_layout(
        title="<b>Healthy diet affordability and economic conditions</b>",
        template="plotly_white", height=620,
        font=dict(family="Arial, Helvetica, sans-serif", color=DARK_GREEN),
        title_font=dict(size=22, color=DARK_GREEN),
        margin=dict(l=75, r=40, t=90, b=80),
        xaxis=dict(title=economic_vars["GDP_PPP"], zeroline=False, range=scatter_x_ranges["GDP_PPP"], autorange=False),
        yaxis=dict(title="Population unable to afford a healthy diet (%)", range=[0, 100], autorange=False),
    )
    scatter_fig.update_layout(font=dict(family="Arial, Helvetica, sans-serif", color=DARK_GREEN, size=14), paper_bgcolor=CREAM, plot_bgcolor=CREAM)
    scatter_fig.update_xaxes(gridcolor="#D7DFD2", linecolor=GREEN)
    scatter_fig.update_yaxes(gridcolor="#D7DFD2", linecolor=GREEN)
    scatter_html = scatter_fig.to_html(include_plotlyjs=False, full_html=False, div_id="scatter-chart")

    country_names = dict(df[["ISO3", "Country"]].dropna().drop_duplicates(subset=["ISO3"]).itertuples(index=False, name=None))
    available_countries = {
        metric: [
            {"iso3": iso3, "name": country_names[iso3], "continent": continent_by_iso.get(iso3, "")}
            for iso3 in sorted(
                df.loc[df[metric].notna() & df["ISO3"].notna(), "ISO3"].unique(),
                key=lambda code: country_names.get(code, code),
            )
            if iso3 in country_names
        ]
        for metric in ("CoHD", "PUA")
    }

    controls = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    body {{ margin: 0; font-family: 'DM Sans', Arial, sans-serif; color: #243047; background: #f5f7fb; }}
    .app-header {{ padding: 28px 36px 10px; }}
    .app-eyebrow {{ margin: 0 0 5px; font-size: 11px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; color: #64748b; }}
    .app-title {{ margin: 0; font-size: 30px; font-weight: 800; letter-spacing: -1px; color: #172554; }}
    .app-subtitle {{ margin: 7px 0 0; font-size: 14px; color: #64748b; }}
    .map-card {{ margin: 12px 26px 26px; border: 1px solid #e2e8f0; border-radius: 18px; background: #fff; overflow: hidden; box-shadow: 0 8px 30px rgba(30,41,59,.06); }}
    .map-content {{ display: grid; grid-template-columns: 240px minmax(0, 1fr); align-items: stretch; }}
    .map-controls {{ display: flex; flex-direction: column; align-items: stretch; gap: 30px; padding: 28px 20px; border-right: 1px solid #e2e8f0; background: #fafbfe; }}
    .map-control {{ display: flex; flex-direction: column; align-items: stretch; gap: 12px; }}
    #indicator-select {{ padding: 9px 12px; border: 1px solid #cbd5e1; border-radius: 7px; font-size: 14px; background: white; }}
    .year-slider-wrap {{ position: relative; width: 100%; box-sizing: border-box; padding-top: 28px; padding-bottom: 18px; }}
    #year-slider {{ display: block; width: 100%; margin: 0; height: 5px; cursor: pointer; accent-color: #2563eb; }}
    #year-value {{ position: absolute; top: 0; left: 0; transform: translateX(-50%); padding: 3px 8px; background: #1d4ed8; color: white; border-radius: 5px; font-size: 13px; font-weight: 700; pointer-events: none; white-space: nowrap; }}
    .year-extremes {{ display: flex; justify-content: space-between; margin-top: 12px; font-size: 12px; color: #64748b; }}
    #world-map {{ min-width: 0; width: 100%; }}
    .app-tabs {{ display: flex; gap: 8px; padding: 8px 26px 0; flex-wrap: wrap; }}
    .tab-button {{ border: 1px solid #dbe3ee; background: white; border-radius: 9px; padding: 12px 17px; font: 700 14px 'DM Sans', Arial, sans-serif; color: #475569; cursor: pointer; }}
    .tab-button.active {{ background: #1d4ed8; border-color: #1d4ed8; color: white; }}
    #map-section[hidden], #trend-section[hidden], #scatter-section[hidden], #food-section[hidden] {{ display: none !important; }}
    #food-section .food-layout {{ display: grid; grid-template-columns: 240px minmax(0, 1fr); align-items: stretch; }}
    #food-section .food-panels {{ min-width: 0; }}
    #food-section .food-panel {{ padding: 20px 25px 12px; }}
    .food-controls {{ display: flex; flex-direction: column; gap: 24px; padding: 28px 20px; background: #fafbfe; border-right: 1px solid #e2e8f0; }}
    .food-controls label {{ display: flex; flex-direction: column; gap: 9px; font-size: 13px; font-weight: 700; min-width: 0; }}
    .food-controls select {{ width: 100%; min-width: 0; box-sizing: border-box; padding: 9px 12px; border: 1px solid #cbd5e1; border-radius: 7px; background: white; font: 14px 'DM Sans',Arial,sans-serif; }}
    .food-controls select[multiple] {{ min-height: 135px; }}
    .food-average-controls {{ display: flex; flex-direction: column; gap: 9px; }}
    .food-average-controls > span {{ font-size: 13px; font-weight: 700; }}
    #food-average-checks {{ max-height: 195px; overflow-y: auto; border: 1px solid #cbd5e1; border-radius: 8px; background: white; padding: 7px; }}

    .food-mode-tabs {{ display:flex; gap:6px; width:100%; }}
    .food-mode-button {{ flex:1; min-width:0; padding:11px 5px; border:1px solid #cbd5e1; border-radius:9px; background:white; color:#475569; font:700 13px 'DM Sans',Arial,sans-serif; cursor:pointer; }}
    .food-mode-button.active {{ background:#1d4ed8; color:white; border-color:#1d4ed8; }}
    .food-country-options {{ max-height:420px; overflow-y:auto; border:1px solid #dbe3ee; border-radius:10px; background:white; padding:7px; }}
    .food-country-options .food-check {{ display:flex; flex-direction:row; align-items:center; gap:8px; padding:6px 4px; font-size:12px; font-weight:500; cursor:pointer; }}
    .food-country-options .food-check input {{ flex-shrink:0; accent-color:#2563eb; }}
    .food-mode-help {{ font-size:12px; line-height:1.55; color:#64748b; }}
    .food-control-actions {{ display:flex; gap:8px; }}
    .food-control-actions button {{ flex:1; padding:7px 5px; border:1px solid #dbe3ee; background:white; border-radius:7px; cursor:pointer; font-size:11px; }}
    #food-country-panel[hidden], #food-continent-panel[hidden] {{ display:none !important; }}
    @media (max-width: 700px) {{ #food-section .food-layout {{ grid-template-columns: 1fr; }} .food-controls {{ border-right: 0; border-bottom: 1px solid #e2e8f0; padding: 18px; }} }}
    #food-parallel-chart {{ width: 100%; min-width: 0; }}
    .food-continent-color-key {{ display:grid; grid-template-columns:1fr; gap:7px; margin-top:10px; padding:10px 12px; border:1px solid #A5B89B; border-radius:10px; }}
    .food-continent-key-item {{ display:flex; align-items:center; gap:9px; font-size:13px; font-weight:500; }}
    .food-continent-key-item i {{ display:inline-block; width:11px; height:11px; border-radius:50%; flex:none; }}

    .food-parallel-details {{ max-height: 350px; overflow-y: auto; box-sizing: border-box; margin: 4px 25px 14px; padding: 16px 20px; background: #f8fafc; border: 1px solid #dbe3ee; border-radius: 12px; font-size: 13px; color: #334155; }}
    .food-parallel-details strong {{ font-size: 15px; color: #172554; }}
    .food-parallel-details table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    .food-parallel-details td {{ padding: 6px 8px; border-bottom: 1px solid #e2e8f0; }}
    .food-parallel-details td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .food-parallel-legend {{ display: flex; flex-wrap: wrap; gap: 10px 18px; padding: 0 25px 15px; font-size: 12px; color: #475569; }}
    .food-parallel-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .food-parallel-legend i {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; }}
    .food-info {{ padding: 4px 24px 12px; color: #475569; font-size: 13px; }}
    .food-note {{ padding: 0 24px 24px; color: #64748b; font-size: 12px; line-height: 1.6; }}
    .scatter-content {{ display: grid; grid-template-columns: 240px minmax(0, 1fr); }}
    #scatter-chart {{ width: 100%; min-width: 0; }}
    .scatter-note {{ padding: 0 26px 20px; font-size: 12px; color: #64748b; line-height: 1.6; }}
    .trend-content {{ display: grid; grid-template-columns: 240px minmax(0, 1fr); align-items: stretch; }}
    .trend-controls {{ display: flex; flex-direction: column; align-items: stretch; gap: 28px; padding: 28px 20px; border-right: 1px solid #e2e8f0; background: #fafbfe; }}
    #trend-chart {{ min-width: 0; width: 100%; }}
    .trend-field {{ display: flex; flex-direction: column; gap: 9px; font-size: 13px; font-weight: 700; }}
    .trend-field select {{ font: 14px 'DM Sans', Arial, sans-serif; border: 1px solid #cbd5e1; border-radius: 7px; padding: 9px; width: 100%; min-width: 0; box-sizing: border-box; background: white; }}
    .country-search {{ width:100%;box-sizing:border-box;padding:9px 11px;border:1px solid #cbd5e1;border-radius:7px;background:white;font:13px 'DM Sans',Arial,sans-serif; }}
    #trend-country-checks, #trend-continent-checks {{ max-height: 205px; overflow-y: auto; border: 1px solid #cbd5e1; border-radius: 8px; background: white; padding: 7px; }}
    .check-row {{ display: flex; align-items: center; gap: 9px; padding: 6px 5px; font-size: 13px; font-weight: 500; cursor: pointer; }}
    .check-row input {{ width: 16px; height: 16px; margin: 0; cursor: pointer; flex-shrink: 0; }}
    .check-row span {{ min-width: 0; overflow-wrap: anywhere; }}
    .check-actions {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .check-actions button {{ font: 600 11px 'DM Sans', Arial, sans-serif; border: 1px solid #cbd5e1; border-radius: 6px; background: white; padding: 6px 8px; cursor: pointer; }}
    .trend-note {{ padding: 0 26px 20px; font-size: 12px; color: #64748b; line-height: 1.6; }}
    @media (max-width: 700px) {{ .map-content {{ grid-template-columns: 1fr; }} .map-controls {{ border-right: 0; border-bottom: 1px solid #e2e8f0; padding: 18px; gap: 18px; }} .year-slider-wrap {{ max-width: 400px; }} }}
    @media (max-width: 700px) {{ .trend-content {{ grid-template-columns: 1fr; }} .trend-controls {{ border-right: 0; border-bottom: 1px solid #e2e8f0; padding: 18px; gap: 18px; }} }}

    :root {{ --cream:#FBFBEC; --dark:#07391E; --green:#4F7E52; --orange:#E5871E; --purple:#8F6688; }}
    body {{ background:var(--cream); color:var(--dark); }}
    .app-header {{ padding-top:32px; }}
    .app-eyebrow {{ color:var(--orange); font-size:12px; }}
    .app-title {{ color:var(--dark); font-size:34px; }}
    .app-subtitle {{ color:var(--green); }}
    .map-card {{ background:var(--cream); border:2px solid var(--green); box-shadow:0 8px 24px rgba(7,57,30,.07); }}
    .map-controls,.trend-controls,.food-controls {{ background:var(--cream); border-color:#A5B89B; color:var(--dark); }}
    .tab-button {{ color:var(--dark); background:var(--cream); border:2px solid var(--green); }}
    .tab-button.active,.food-mode-button.active {{ background:var(--green); border-color:var(--green); color:var(--cream); }}
    .food-mode-button {{ background:var(--cream); border-color:var(--green); color:var(--dark); }}
    #indicator-select,.trend-field select,.food-controls select,.country-search {{ background:var(--cream); color:var(--dark); border:1px solid var(--green); }}
    #year-slider,#scatter-year,.check-row input,.food-country-options .food-check input {{ accent-color:var(--orange); }}
    #year-value {{ background:var(--green); color:var(--cream); }}
    .year-extremes,.food-mode-help,.food-info,.food-note,.trend-note,.scatter-note {{ color:var(--dark); }}
    #trend-country-checks,#trend-continent-checks,#food-average-checks,.food-country-options {{ background:var(--cream); border-color:#A5B89B; color:var(--dark); }}
    .check-actions button,.food-control-actions button {{ background:var(--cream); color:var(--dark); border-color:var(--green); }}
    .food-parallel-details {{ background:var(--cream); color:var(--dark); border-color:var(--green); }}
    .food-parallel-details strong {{ color:var(--dark); }}
    .food-parallel-details td {{ border-color:#A5B89B; }}
    .food-parallel-legend {{ color:var(--dark); }}

    /* GREEN OUTER SHELL · CREAM DATA PANELS */
    :root {{ --cream:#FBFBEC; --dark:#07391E; --green:#4F7E52; --orange:#E5871E; --purple:#8F6688; }}
    body {{ background:var(--green); color:var(--cream); }}
    .app-header {{ padding:36px 38px 18px; }}
    .app-eyebrow {{ color:var(--orange); font-size:13px; letter-spacing:2.5px; }}
    .app-title {{ color:var(--cream); font-size:38px; font-weight:800; }}
    .app-subtitle {{ color:var(--cream); opacity:.95; font-size:15px; }}
    .app-tabs {{ gap:12px; padding:10px 28px 16px; }}
    .tab-button {{ background:var(--cream); color:var(--dark); border:2px solid var(--cream); border-radius:12px; padding:13px 20px; font-weight:700; transition:background .2s ease,color .2s ease; }}
    .tab-button:hover:not(.active) {{ background:var(--cream); color:var(--dark); border-color:var(--cream); }}
    .tab-button.active {{ background:var(--orange); color:var(--dark); border-color:var(--orange); }}
    .map-card {{ background:var(--cream); border:2px solid var(--cream); border-radius:20px; box-shadow:0 12px 32px rgba(7,57,30,.18); overflow:hidden; }}
    .map-controls,.trend-controls,.food-controls {{ background:var(--cream); color:var(--dark); border-color:#A5B89B; }}
    .map-control,.trend-field,.food-controls label {{ color:var(--dark); }}
    #indicator-select,.trend-field select,.food-controls select,.country-search {{ background:var(--cream); color:var(--dark); border:1.5px solid var(--green); border-radius:8px; }}
    #year-slider,#scatter-year,.check-row input,.food-country-options .food-check input {{ accent-color:var(--orange); }}
    #year-value {{ background:var(--dark); color:var(--cream); }}
    .check-actions button,.food-control-actions button {{ background:var(--cream); color:var(--dark); border:1px solid var(--green); border-radius:7px; }}
    .check-actions button:hover,.food-control-actions button:hover {{ background:var(--orange); color:var(--dark); }}
    .food-mode-button {{ background:var(--cream); color:var(--dark); border:1.5px solid var(--green); }}
    .food-mode-button.active {{ background:var(--orange); color:var(--dark); border-color:var(--orange); }}
    #trend-country-checks,#trend-continent-checks,#food-average-checks,.food-country-options {{ background:var(--cream); color:var(--dark); border-color:#A5B89B; }}
    .food-parallel-details {{ background:var(--cream); color:var(--dark); border-color:var(--green); }}
    .food-parallel-details strong,.trend-note,.scatter-note,.food-note,.food-info,.food-mode-help,.year-extremes,.food-parallel-legend {{ color:var(--dark); }}
    .food-parallel-details td {{ border-color:#A5B89B; }}
    .trend-background-toggle {{ display:flex; align-items:flex-start; gap:9px; cursor:pointer; line-height:1.45; }}
    .trend-background-toggle input {{ margin-top:3px; accent-color:var(--orange); flex-shrink:0; }}
    </style>
    <div class="app-header">
      <p class="app-eyebrow">DATA EXPLORER · 2017–2025</p>
      <h1 class="app-title">Is Healthy Eating a Luxury?</h1>
      <p class="app-subtitle">Explore the cost and affordability of a healthy diet worldwide.</p>
    </div>
    <nav class="app-tabs" aria-label="Dashboard sections">
      <button type="button" class="tab-button active" id="tab-map">Geographical Distribution</button>
      <button type="button" class="tab-button" id="tab-trend">Temporal Evolution</button>
      <button type="button" class="tab-button" id="tab-scatter">Economic Relationships</button>
      <button type="button" class="tab-button" id="tab-food">Food Cost Composition</button>
    </nav>
    <div id="map-section" class="map-card">
    <div class="map-content">
    <div class="map-controls">
      <div class="map-control">
        <label for="indicator-select">Indicator</label>
        <select id="indicator-select">
          <option value="PUA" selected>PUA · Unaffordability (%)</option>
          <option value="CoHD">CoHD · Daily diet cost (PPP $)</option>
        </select>
      </div>
      <div class="map-control">
        <label for="year-slider">Year</label>
        <div class="year-slider-wrap">
          <output id="year-value" for="year-slider">{initial_key[1]}</output>
          <input type="range" id="year-slider" min="{min(years)}" max="{max(years)}" step="1" value="{initial_key[1]}">
          <div class="year-extremes"><span>{min(years)}</span><span>{max(years)}</span></div>
        </div>
      </div>
    </div>
    """

    trend_section = f"""
    <section id="trend-section" class="map-card" hidden>
      <div class="trend-content">
        <div class="trend-controls">
          <label class="trend-field" for="trend-indicator">Indicator
            <select id="trend-indicator">
              <option value="PUA" selected>PUA · Unaffordability (%)</option>
              <option value="CoHD">CoHD · Daily diet cost (PPP $)</option>
            </select>
          </label>
          <label class="trend-field" for="trend-average">Show world average
            <input id="trend-average" type="checkbox" checked>
          </label>
          <div class="trend-field">
            <span>Continent averages</span>
            <div class="check-actions"><button type="button" id="continents-all">Select all</button><button type="button" id="continents-clear">Clear selection</button></div>
            <div id="trend-continent-checks" role="group" aria-label="Select continent averages"></div>
          </div>
          <div class="trend-field">
            <label class="trend-background-toggle" for="trend-background-visible"><input id="trend-background-visible" type="checkbox" checked> Show all country trajectories in the background</label>
          </div>
          <div class="trend-field">
            <span>Countries to compare</span>
            <div class="check-actions"><button type="button" id="countries-clear">Clear all</button></div>
            <input id="trend-country-search" class="country-search" type="search" placeholder="Search for a country..." aria-label="Search for a country">
            <div id="trend-country-checks" role="group" aria-label="Select countries"></div>
          </div>
        </div>
        {trend_html}
      </div>
    </section>
    """

    scatter_section = f"""
    <section id="scatter-section" class="map-card" hidden>
      <div class="scatter-content">
        <div class="trend-controls">
          <label class="trend-field" for="scatter-variable">Economic indicator
            <select id="scatter-variable">
              {''.join(f'<option value="{key}">{label}</option>' for key, label in economic_vars.items())}
            </select>
          </label>
          <label class="trend-field" for="scatter-year">Year
            <input id="scatter-year" type="range" min="{min(years)}" max="{max(years)}" value="{max(years)}" step="1">
            <output id="scatter-year-value">{max(years)}</output>
          </label>
          <div class="trend-field"><strong>Continents</strong>
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0">
              <button type="button" id="scatter-select-all">Select all</button>
              <button type="button" id="scatter-clear-all">Clear selection</button>
            </div>
            <div id="scatter-continent-list" style="display:flex;flex-direction:column;gap:9px;padding:12px;border:1px solid #A7BFA2;border-radius:10px"></div>
          </div>
          <div id="scatter-stats" class="trend-note" style="padding:0"></div>
        </div>
        {scatter_html}
      </div>
    </section>
    """

    food_section = f"""
    <section id="food-section" class="map-card" hidden>
      <div class="food-layout">
        <aside class="food-controls">
          <div class="trend-field"><strong>Filter by continent</strong>
            <select id="food-continent-filter" aria-label="Filter food profiles by continent">
              <option value="">All continents</option>
              {''.join(f'<option value="{c}">{c}</option>' for c in ['Africa','Asia','Europe','North America','South America','Oceania'])}
            </select>
          </div>
          <label class="trend-background-toggle"><input type="checkbox" id="food-background-toggle" checked>Show all country profiles in the background</label>
          <div class="trend-field"><strong>Countries to compare</strong>
            <div class="check-actions"><button type="button" id="food-countries-clear">Clear all</button></div>
            <input id="food-country-search" class="country-search" type="search" placeholder="Search for a country..." aria-label="Search for a country">
            <div id="food-country-options" class="food-country-options" role="group" aria-label="Select countries"></div>
          </div>
        </aside>
        <div class="food-panels">
          <div class="food-panel">
            {food_parallel_html}
            <div id="food-parallel-details" class="food-parallel-details" aria-live="polite">Click a country line to select it and see its six food category values.</div>
            <p class="food-note" id="food-parallel-info"></p>
          </div>
        </div>
      </div>
    </section>
    """

    script = f"""
    <script>
    (function () {{
      const foodData = {json.dumps(food_records, ensure_ascii=False)};
      const foodLabels = {json.dumps(food_categories, ensure_ascii=False)};
      const foodSection = document.getElementById('food-section');
      const tabFood = document.getElementById('tab-food');
      const foodParallelChart = document.getElementById('food-parallel-chart');
      const foodParallelDetails = document.getElementById('food-parallel-details');
      const foodCountryOptions = document.getElementById('food-country-options');
      const continentPalette = {json.dumps(CONTINENT_COLORS, ensure_ascii=False)};
      const foodKeys = Object.keys(foodLabels);
      const foodFiveContinents = ['Africa','Asia','Europe','North America','South America'];
      const foodAvailable = foodData.filter(r => foodKeys.every(k =>
        r.categories[k] !== null && Number.isFinite(r.categories[k])));
      // Scala identica per tutte le categorie, in PPP $ per persona al giorno.
      const foodParallelMax = Math.max(0.01, ...foodAvailable.flatMap(r =>
        foodKeys.map(k => r.categories[k]))) * 1.05;
      const foodCountrySelection = new Set();
      const foodHighlighted = new Set();
      const foodContinentFilter = document.getElementById('food-continent-filter');
      const foodBackgroundToggle = document.getElementById('food-background-toggle');

      function foodRows() {{
        const continent=foodContinentFilter.value;
        return foodAvailable.filter(r =>
          continent ? r.continent===continent : foodCountrySelection.has(r.iso3));
      }}
      function refreshFoodCountryOptions() {{
        foodCountryOptions.replaceChildren();
        const search=document.getElementById('food-country-search').value.trim().toLocaleLowerCase('it');
        const sorted = [...foodAvailable].sort((a,b)=>a.country.localeCompare(b.country,'it'));
        for (const r of sorted) {{
          if(!r.country.toLocaleLowerCase('it').includes(search)) continue;
          const label=document.createElement('label');
          label.className='food-check';
          const box=document.createElement('input');
          box.type='checkbox'; box.value=r.iso3;
          box.checked=foodCountrySelection.has(r.iso3);
          box.addEventListener('change',()=>{{
            if(box.checked) foodCountrySelection.add(r.iso3);
            else {{foodCountrySelection.delete(r.iso3);foodHighlighted.delete(r.iso3);}}
            updateFoodParallel();
          }});
          const dot=document.createElement('span');
          dot.style.cssText='display:inline-block;width:9px;height:9px;border-radius:50%;flex-shrink:0';
          dot.style.backgroundColor=continentPalette[r.continent]||'#8F6688';
          label.append(box,dot,document.createTextNode(r.country));
          foodCountryOptions.append(label);
        }}
      }}
      function showFoodParallelDetails(rows) {{
        const chosen=rows.filter(r=>foodHighlighted.has(r.iso3));
        foodParallelDetails.replaceChildren();
        if (!chosen.length) {{
          foodParallelDetails.textContent='Click any country line to select it and view its food costs. Click again to deselect.';
          return;
        }}
        for(const r of chosen) {{
          const card=document.createElement('div');card.style.marginBottom='15px';
          const heading=document.createElement('strong');
          heading.textContent=r.country+(r.isAverage?' · '+r.count+' countries':' · '+r.continent);
          heading.style.color=continentPalette[r.continent]||'#334155';
          card.append(heading);
          const table=document.createElement('table');
          let total=0;
          for(const k of foodKeys) {{
            const tr=document.createElement('tr');
            const name=document.createElement('td');name.textContent=foodLabels[k];
            const value=document.createElement('td');value.textContent=r.categories[k].toFixed(3)+' PPP $';
            total+=r.categories[k];tr.append(name,value);table.append(tr);
          }}
          const tr=document.createElement('tr');
          const name=document.createElement('td');name.textContent='Sum of categories';name.style.fontWeight='700';
          const value=document.createElement('td');value.textContent=total.toFixed(3)+' PPP $';value.style.fontWeight='700';
          tr.append(name,value);table.append(tr);
          if(r.cohd!==null) {{
            const extra=document.createElement('tr');
            const label=document.createElement('td');label.textContent='CoHD in dataset';
            const val=document.createElement('td');val.textContent=r.cohd.toFixed(3)+' PPP $';
            extra.append(label,val);table.append(extra);
          }}
          card.append(table);foodParallelDetails.append(card);
        }}
        const clear=document.createElement('button');clear.type='button';
        clear.textContent='Clear highlight';
        clear.style.cssText='padding:7px 12px;border:1px solid #cbd5e1;border-radius:7px;background:white;cursor:pointer';
        clear.addEventListener('click',()=>{{foodHighlighted.clear();updateFoodParallel();}});
        foodParallelDetails.append(clear);
      }}
      function updateFoodParallel() {{
        const rows=foodRows();
        const background=foodBackgroundToggle.checked
          ? foodAvailable.filter(r=>!foodCountrySelection.has(r.iso3) && !foodContinentFilter.value) : [];
        const visibleIds=new Set(rows.map(r=>r.iso3));
        for(const iso of [...foodHighlighted]) if(!visibleIds.has(iso)) foodHighlighted.delete(iso);
        const ordered=[...background,...rows.filter(r=>!foodHighlighted.has(r.iso3)),
                       ...rows.filter(r=>foodHighlighted.has(r.iso3))];
        const x=foodKeys.map((_,i)=>i);
        const traces=ordered.map(r=>{{
          const selected=foodHighlighted.has(r.iso3);
          const isBackground=!visibleIds.has(r.iso3);
          const color=continentPalette[r.continent]||'#8F6688';
          return {{type:'scatter',mode:isBackground?'lines':'lines+markers',x,
            y:foodKeys.map(k=>r.categories[k]),
            name:r.country,showlegend:false,
            customdata:foodKeys.map(k=>[r.iso3,r.country,r.continent,foodLabels[k],r.categories[k]]),
            line:{{color,width:selected?4.5:isBackground?0.8:2.8,
              dash:'solid'}},
            marker:{{size:selected?9:7,color}},
            opacity:isBackground?0.09:selected?1:foodHighlighted.size?0.45:0.95,
            hoverinfo:isBackground?'none':undefined,
            hovertemplate:isBackground?undefined:'<b>%{{customdata[1]}}</b> · %{{customdata[2]}}<br>%{{customdata[3]}}: %{{customdata[4]:.3f}} PPP $<extra></extra>'}};
        }});
        // Una sola voce per continente nella legenda, non una voce per ogni Paese.
        const legendContinents=['Africa','Asia','Europe','North America','South America','Oceania'];
        legendContinents.forEach(c=>{{
          traces.push({{type:'scatter',mode:'lines+markers',x:[null],y:[null],
            name:c,showlegend:true,hoverinfo:'skip',
            line:{{color:continentPalette[c],width:3}},
            marker:{{color:continentPalette[c],size:8}}}});
        }});
        const shapes=foodKeys.map((_,i)=>({{type:'line',xref:'x',yref:'paper',x0:i,x1:i,y0:0,y1:1,
          line:{{color:'#A5B89B',width:1.4}},layer:'below'}}));
        const annotations=foodKeys.map((k,i)=>({{x:i,y:1.04,xref:'x',yref:'paper',showarrow:false,
          text:foodLabels[k],font:{{size:11,color:'#07391E'}},xanchor:'center',yanchor:'bottom'}}));
        const layout={{title:{{text:'<b>Food cost profiles · 2021</b>',font:{{size:21}}}},
          template:'plotly_white',font:{{color:'#07391E',size:14}},height:590,margin:{{l:50,r:35,t:105,b:115}},
          xaxis:{{range:[-0.25,foodKeys.length-0.75],tickmode:'array',tickvals:x,
            ticktext:foodKeys.map(()=>'0 – '+foodParallelMax.toFixed(2)+' PPP $'),
            tickfont:{{size:10}},showgrid:false,zeroline:false,fixedrange:true}},
          yaxis:{{range:[0,foodParallelMax],visible:false,fixedrange:true}},shapes,annotations,
          hovermode:'closest',showlegend:true,legend:{{orientation:'h',x:0.5,xanchor:'center',y:-0.14,yanchor:'top',font:{{size:12,color:'#07391E'}},bgcolor:'rgba(0,0,0,0)',itemclick:false,itemdoubleclick:false}},clickmode:'event',paper_bgcolor:'#FBFBEC',plot_bgcolor:'#FBFBEC'}};
        Plotly.react(foodParallelChart,traces,layout,{{responsive:true,displaylogo:false,scrollZoom:false}});
        showFoodParallelDetails(rows);
        document.getElementById('food-parallel-info').textContent=
          (foodContinentFilter.value ? 'Showing all countries in '+foodContinentFilter.value+'. ' : 'Showing selected countries. ')+
          foodCountrySelection.size+' individually selected countries. '+
          (foodBackgroundToggle.checked && !foodContinentFilter.value ? 'Other countries appear faintly in the background. ' : '')+
          'Click any visible country line to select or deselect it.';
      }}
      // Clicking a line toggles the corresponding country or continent average.
      // Background trajectories are clickable too, even if not yet checked in the sidebar.
      foodParallelChart.on('plotly_click',event=>{{
        const point=event.points&&event.points[0];
        if(!point||!point.customdata)return;
        const iso=point.customdata[0];
        if(foodCountrySelection.has(iso)){{
          foodCountrySelection.delete(iso);
          foodHighlighted.delete(iso);
        }}else{{
          foodCountrySelection.add(iso);
          foodHighlighted.add(iso);
        }}
        refreshFoodCountryOptions();
        updateFoodParallel();
      }});
      foodContinentFilter.addEventListener('change',()=>{{
        foodHighlighted.clear();
        updateFoodParallel();
      }});
      document.getElementById('food-countries-clear').addEventListener('click',()=>{{
        foodCountrySelection.clear();foodHighlighted.clear();refreshFoodCountryOptions();updateFoodParallel();
      }});
      document.getElementById('food-country-search').addEventListener('input',refreshFoodCountryOptions);
      foodBackgroundToggle.addEventListener('change',updateFoodParallel);
      refreshFoodCountryOptions();updateFoodParallel();
      const scatterData = {json.dumps(scatter_data, ensure_ascii=False)};
      const scatterXRanges = {json.dumps(scatter_x_ranges)};
      const economicLabels = {json.dumps(economic_vars, ensure_ascii=False)};
      const scatterPlot = document.getElementById('scatter-chart');
      const scatterSection = document.getElementById('scatter-section');
      const scatterVariable = document.getElementById('scatter-variable');
      const scatterYear = document.getElementById('scatter-year');
      const scatterContinentList = document.getElementById('scatter-continent-list');
      const scatterContinents = {json.dumps(list(continents), ensure_ascii=False)};
      for (const continent of scatterContinents) {{
        const label = document.createElement('label');
        label.style.cssText = 'display:flex;align-items:center;gap:10px;cursor:pointer;font-weight:500';
        const box = document.createElement('input');
        box.type='checkbox';box.value=continent;box.checked=true;box.className='scatter-continent-check';
        box.addEventListener('change',updateScatter);
        label.append(box,document.createTextNode(continent));
        scatterContinentList.append(label);
      }}
      document.getElementById('scatter-select-all').addEventListener('click',()=>{{
        scatterContinentList.querySelectorAll('input').forEach(box=>box.checked=true);updateScatter();
      }});
      document.getElementById('scatter-clear-all').addEventListener('click',()=>{{
        scatterContinentList.querySelectorAll('input').forEach(box=>box.checked=false);updateScatter();
      }});
      const scatterStats = document.getElementById('scatter-stats');
      const continentColors = {json.dumps({**CONTINENT_COLORS, "Unclassified": PURPLE}, ensure_ascii=False)};
      function updateScatter() {{
        const variable = scatterVariable.value, year = Number(scatterYear.value);
        document.getElementById('scatter-year-value').textContent = year;
        const rows = scatterData[variable][year] || [];
        const selectedContinents = new Set([...scatterContinentList.querySelectorAll('input:checked')].map(box=>box.value));
        const filtered = rows.filter(row => selectedContinents.has(row.continent));
        const x = filtered.map(row => row.x), y = filtered.map(row => row.y);
        const continentNames = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania', 'Antarctica', 'Unclassified'];
        const traces = continentNames.filter(continent => filtered.some(row => row.continent === continent)).map(continent => {{
          const group = filtered.filter(row => row.continent === continent);
          return {{
            x: group.map(row => row.x), y: group.map(row => row.y),
            customdata: group.map(row => row.country), mode:'markers', type:'scatter',
            marker:{{size:11, color:continentColors[continent], opacity:.75}},
            name:continent,
            hovertemplate:'<b>%{{customdata}}</b><br>Continent: '+continent+'<br>'+economicLabels[variable]+': %{{x:,.2f}}<br>PUA: %{{y:.2f}}%<extra></extra>'
          }};
        }});
        const n = filtered.length;
        let description = n + ' countries with available data';
        if (n >= 3) {{
          const mx=x.reduce((a,b)=>a+b,0)/n, my=y.reduce((a,b)=>a+b,0)/n;
          const cov=x.reduce((a,v,i)=>a+(v-mx)*(y[i]-my),0);
          const vx=x.reduce((a,v)=>a+(v-mx)**2,0), vy=y.reduce((a,v)=>a+(v-my)**2,0);
          const r = vx>0 && vy>0 ? cov/Math.sqrt(vx*vy) : null;
          if (r !== null) description += '<br>Pearson correlation: <b>'+r.toFixed(3)+'</b>';
        }}
        scatterStats.innerHTML=description;
        Plotly.react(scatterPlot,traces,{{...scatterPlot.layout,
          title:{{text:'<b>Affordability and '+economicLabels[variable]+' · '+year+'</b>'}},
          xaxis:{{title:{{text:economicLabels[variable],standoff:18,font:{{size:14}}}},zeroline:false,range:scatterXRanges[variable],autorange:false}},
          yaxis:{{title:{{text:'Population unable to afford a healthy diet (%)',standoff:15,font:{{size:14}}}},range:[0,100],autorange:false}},
          showlegend:true, legend:{{orientation:'h', y:-0.23, x:0, title:{{text:'Continent'}}}}
        }},{{responsive:true,displaylogo:false}});
      }}
      scatterVariable.addEventListener('change',updateScatter);
      scatterYear.addEventListener('input',updateScatter);
      const plot = document.getElementById('world-map');
      const indicator = document.getElementById('indicator-select');
      const slider = document.getElementById('year-slider');
      const yearValue = document.getElementById('year-value');
      const years = {years};
      const labels = {{
        CoHD: 'Daily cost of a healthy diet (PPP $)',
        PUA: 'Population unable to afford a healthy diet (%)'
      }};
      let updating = Promise.resolve();
      function positionBubble() {{
        const min = Number(slider.min), max = Number(slider.max);
        const fraction = (Number(slider.value) - min) / (max - min || 1);
        const thumbWidth = 16;
        const position = fraction * (slider.clientWidth - thumbWidth) + thumbWidth / 2;
        yearValue.style.left = position + 'px';
        yearValue.textContent = slider.value;
      }}
      function updateMap() {{
        positionBubble();
        if (!plot || !plot.data || !window.Plotly) return;
        const metric = indicator.value;
        const year = Number(slider.value);
        const yearIndex = years.indexOf(year);
        if (yearIndex === -1) return;
        const selectedIndex = (metric === 'CoHD' ? 0 : 1) * years.length + yearIndex;
        const visible = plot.data.map((_, index) => index === selectedIndex);
        updating = updating.catch(() => {{}}).then(() =>
          Plotly.restyle(plot, {{visible: visible}}).then(() =>
            Plotly.relayout(plot, {{'title.text': '<b>' + labels[metric] + ' — ' + year + '</b>'}})
          )
        );
      }}
      indicator.addEventListener('change', updateMap);
      slider.addEventListener('input', updateMap);
      window.addEventListener('resize', positionBubble);
      positionBubble();

      const mapSection = document.getElementById('map-section');
      const trendSection = document.getElementById('trend-section');
      const tabMap = document.getElementById('tab-map');
      const tabTrend = document.getElementById('tab-trend');
      const trendPlot = document.getElementById('trend-chart');
      const trendIndicator = document.getElementById('trend-indicator');
      const trendCountryChecks = document.getElementById('trend-country-checks');
      const trendAverage = document.getElementById('trend-average');
      const trendContinentChecks = document.getElementById('trend-continent-checks');
      const trendKeys = {json.dumps(trend_keys, ensure_ascii=False)};
      const trendAxisRanges = {json.dumps(trend_axis_ranges)};
      const availableCountries = {json.dumps(available_countries, ensure_ascii=False)};
      const continentColorsTrend = {json.dumps(CONTINENT_COLORS, ensure_ascii=False)};
      const continentsTrend = {json.dumps(list(continents), ensure_ascii=False)};
      const selectedCountries = new Set();
      const backgroundVisible = document.getElementById('trend-background-visible');
      const selectedContinents = new Set();
      function addCheckbox(container, value, label, selected, color, onChange) {{
        const row = document.createElement('label');
        row.className = 'check-row';
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.value = value;
        box.checked = selected;
        box.style.accentColor = color || '#2563eb';
        const caption = document.createElement('span');
        caption.textContent = label;
        row.append(box, caption);
        container.appendChild(row);
        box.addEventListener('change', () => onChange(box.checked));
      }}
      function refreshCountryOptions() {{
        const metric = trendIndicator.value;
        const search=document.getElementById('trend-country-search').value.trim().toLocaleLowerCase('it');
        trendCountryChecks.replaceChildren();
        for (const country of availableCountries[metric]) {{
          if(!country.name.toLocaleLowerCase('it').includes(search)) continue;
          addCheckbox(trendCountryChecks, country.iso3, country.name, selectedCountries.has(country.iso3),
            continentColorsTrend[country.continent], checked => {{
              if (checked) selectedCountries.add(country.iso3);
              else selectedCountries.delete(country.iso3);
              updateTrend();
            }});
        }}
      }}
      function refreshContinentOptions() {{
        trendContinentChecks.replaceChildren();
        for (const continent of continentsTrend) {{
          addCheckbox(trendContinentChecks, continent, continent, selectedContinents.has(continent),
            continentColorsTrend[continent], checked => {{
              if (checked) selectedContinents.add(continent);
              else selectedContinents.delete(continent);
              updateTrend();
            }});
        }}
      }}
      function updateTrend() {{
        const metric = trendIndicator.value;
        const visible = [];
        const opacities = [];
        const lineWidths = [];
        const markerSizes = [];
        const legendVisibility = [];
        trendKeys.forEach(([m, country]) => {{
          if (m !== metric) {{
            legendVisibility.push(false);
            visible.push(false); opacities.push(1); lineWidths.push(2); markerSizes.push(6);
          }} else if (country === '__average__') {{
            legendVisibility.push(false);
            visible.push(trendAverage.checked); opacities.push(1); lineWidths.push(4); markerSizes.push(9);
          }} else if (country.startsWith('__continent__:')) {{
            legendVisibility.push(false);
            visible.push(selectedContinents.has(country.slice('__continent__:'.length)));
            opacities.push(1); lineWidths.push(3); markerSizes.push(9);
          }} else {{
            const selected = selectedCountries.has(country);
            legendVisibility.push(selected);
            visible.push(selected || backgroundVisible.checked);
            opacities.push(selected ? 1 : 0.12);
            lineWidths.push(selected ? 3.5 : 1.15);
            markerSizes.push(selected ? 7 : 2);
          }}
        }});
        // Preserve the traces so clicking a background trajectory can select its country.
        Plotly.restyle(trendPlot, {{
          visible: visible,
          opacity: opacities,
          'line.width': lineWidths,
          'marker.size': markerSizes,
          showlegend: legendVisibility
        }});
        Plotly.relayout(trendPlot, {{
          'title.text': '<b>' + (metric === 'PUA' ? 'Unaffordability over time' : 'Daily diet cost over time') + ' · 2017–2025</b>',
          'yaxis.title.text': metric === 'PUA' ? 'Population unable to afford a healthy diet (%)' : 'Daily cost per person (PPP $)',
          'yaxis.range': trendAxisRanges[metric],
          'yaxis.autorange': false,
          showlegend: legendVisibility.some(Boolean)
        }});
      }}
      // Clicking any country's line toggles its selection, like the parallel plot.
      trendPlot.on('plotly_click', event => {{
        if (!event.points || !event.points.length) return;
        const index = event.points[0].curveNumber;
        const entry = trendKeys[index];
        if (!entry || entry[0] !== trendIndicator.value) return;
        const iso3 = entry[1];
        if (iso3 === '__average__' || iso3.startsWith('__continent__:')) return;
        if (selectedCountries.has(iso3)) selectedCountries.delete(iso3);
        else selectedCountries.add(iso3);
        refreshCountryOptions();
        updateTrend();
      }});
      backgroundVisible.addEventListener('change', updateTrend);
      trendIndicator.addEventListener('change', () => {{
        refreshCountryOptions();
        updateTrend();
      }});
      trendAverage.addEventListener('change', updateTrend);
      document.getElementById('countries-clear').addEventListener('click', () => {{
        selectedCountries.clear(); refreshCountryOptions(); updateTrend();
      }});
      document.getElementById('continents-all').addEventListener('click', () => {{
        continentsTrend.forEach(c => selectedContinents.add(c)); refreshContinentOptions(); updateTrend();
      }});
      document.getElementById('continents-clear').addEventListener('click', () => {{
        selectedContinents.clear(); refreshContinentOptions(); updateTrend();
      }});
      refreshContinentOptions();
      document.getElementById('trend-country-search').addEventListener('input',refreshCountryOptions);
      const tabScatter = document.getElementById('tab-scatter');
      function showSection(section) {{
        mapSection.hidden = section !== 'map';
        trendSection.hidden = section !== 'trend';
        scatterSection.hidden = section !== 'scatter';
        foodSection.hidden = section !== 'food';
        tabMap.classList.toggle('active', section === 'map');
        tabTrend.classList.toggle('active', section === 'trend');
        tabScatter.classList.toggle('active', section === 'scatter');
        tabFood.classList.toggle('active', section === 'food');
        if (section === 'trend') {{ updateTrend(); Plotly.Plots.resize(trendPlot); }}
        else if (section === 'scatter') {{ updateScatter(); Plotly.Plots.resize(scatterPlot); }}
        else if (section === 'food') {{ updateFoodParallel(); Plotly.Plots.resize(foodParallelChart); }}
        else Plotly.Plots.resize(plot);
      }}
      refreshCountryOptions();
      tabMap.addEventListener('click', () => showSection('map'));
      tabTrend.addEventListener('click', () => showSection('trend'));
      tabScatter.addEventListener('click', () => showSection('scatter'));
      tabFood.addEventListener('click', () => showSection('food'));
    }})();
    </script>
    """

    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"</head><body>{controls}{plot_html}</div></div>{trend_section}{scatter_section}{food_section}{script}</body></html>"
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"\nInteractive dashboard saved to: {output_path}")

    return output_path


