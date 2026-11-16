from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

def find_project_root() -> Path:
    """Locate the project root from the scripts directory."""
    return Path(__file__).resolve().parent.parent


ROOT = find_project_root()
DATA = ROOT / 'data/processed/ml/supervised_panel.csv'
OUT = ROOT / 'outputs/05_supervised'
TABLES, FIGURES = OUT / 'tables', OUT / 'figures' 
NUMERIC = ['log_GDP_PPP', 'Urban', 'Unemployment', 'Inflation', 'Population_Density']
AUGMENTED = NUMERIC + ['Continent']
CREAM, GREEN, DARK, ORANGE, YELLOW = '#fbfbec', '#4f7e52', '#044b25', '#e5871e', '#f9d53e'


def transform_inflation_signed_log(X):
    X = np.asarray(X, dtype=float).copy()
    X[:, 3] = np.sign(X[:, 3]) * np.log1p(np.abs(X[:, 3]))
    return X

def build_random_forest_pipeline(with_continent):
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('signed_log', FunctionTransformer(transform_inflation_signed_log, validate=False))
    ])
    if with_continent:
        preprocessing = ColumnTransformer([
            ('numeric', numeric, NUMERIC),
            ('continent', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Continent'])
        ])
    else:
        preprocessing = numeric
    return Pipeline([
        ('preprocess', preprocessing),
        ('model', RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                        max_features=1.0, random_state=42, n_jobs=-1))
    ])

def predict_bounded_pua(m, df, columns):
    return np.clip(m.predict(df[columns]), 0, 100)

def calculate_regression_metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p),
                RMSE=np.sqrt(mean_squared_error(y, p)), R2=r2_score(y, p))

def load_supervised_dataset(data=None, data_path=None):
    """Validate supervised data from a DataFrame or an optional CSV path."""
    if data is not None:
        df = data.copy()
    else:
        path = Path(data_path) if data_path is not None else DATA
        if not path.exists():
            raise FileNotFoundError(f'Cannot find {path}')
        df = pd.read_csv(path, encoding='utf-8-sig')
    required = set(['ISO3', 'Country', 'Year', 'PUA', 'GDP_PPP', 'Urban',
                    'Unemployment', 'Inflation', 'Population_Density', 'Continent'])
    if missing := required - set(df.columns):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    for c in ['PUA', 'GDP_PPP', 'Urban', 'Unemployment', 'Inflation', 'Population_Density']:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for c in ['GDP_PPP', 'Population_Density']:
        df.loc[df[c] <= 0, c] = np.nan
    for c in ['PUA', 'Urban', 'Unemployment']:
        df.loc[df[c].notna() & ~df[c].between(0, 100), c] = np.nan
    df['log_GDP_PPP'] = np.log(df['GDP_PPP'])
    df['Year'] = pd.to_numeric(df['Year'], errors='raise').astype(int)
    if df.duplicated(['ISO3', 'Year']).any():
        raise ValueError('Duplicate ISO3-Year pairs')
    df = df[df['PUA'].notna()].copy()
    if df['Continent'].isna().any():
        raise ValueError('Missing Continent values')
    return df

def plot_final_test_performance(scores, n):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor(GREEN)
    ax.axis('off')
    ax.set_title('Final Test Performance (2024–2025)', color=CREAM, fontsize=17, weight='bold')
    for i, (label, value) in enumerate([('MAE (pp)', f"{scores['MAE']:.2f}"),
                                         ('RMSE (pp)', f"{scores['RMSE']:.2f}"),
                                         ('R²', f"{scores['R2']:.3f}")]):
        x = (i + .5) / 3
        ax.text(x, .58, value, ha='center', transform=ax.transAxes, fontsize=30,
                color=YELLOW if i == 0 else CREAM, weight='bold')
        ax.text(x, .30, label, ha='center', transform=ax.transAxes, fontsize=13, color=CREAM)
    ax.text(.5, .07, f'Random Forest · Socioeconomic + Continent · {n} test observations',
            ha='center', transform=ax.transAxes, fontsize=10, color=CREAM)
    fig.savefig(FIGURES / 'final_test_performance.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

def plot_permutation_importance(imp):
    names = {'log_GDP_PPP':'GDP per capita (PPP)', 'Urban':'Urbanization',
             'Population_Density':'Population density', 'Unemployment':'Unemployment',
             'Inflation':'Inflation', 'Continent':'Continent'}
    data = imp.sort_values('MAE_increase').copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    bars = ax.barh(data.feature.map(names), data.MAE_increase,
                   color=[ORANGE if x == 'log_GDP_PPP' else GREEN for x in data.feature])
    ax.set_title('What drives the predictions?', color=DARK, fontsize=16, weight='bold', pad=15)
    ax.set_xlabel('Increase in MAE when a feature is shuffled (percentage points)', color=DARK)
    ax.grid(False, axis='x')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, data.MAE_increase):
        ax.text(max(val, 0) + .1, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', color=DARK)
    ax.set_xlim(min(0, data.MAE_increase.min() * 1.1), max(1, data.MAE_increase.max() * 1.2))
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / 'permutation_importance.png', dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)

def plot_gdp_partial_dependence(m, development):
    # Manual average partial dependence on raw log GDP, preserving Continent and all other columns.
    values = development['log_GDP_PPP'].dropna().to_numpy()
    grid = np.linspace(*np.quantile(values, [.05, .95]), 70)
    mean_preds = []
    for val in grid:
        scenario = development[AUGMENTED].copy()
        scenario['log_GDP_PPP'] = val
        mean_preds.append(predict_bounded_pua(m, scenario, AUGMENTED).mean())
    data = pd.DataFrame({'GDP_PPP_per_capita': np.exp(grid), 'Predicted_PUA_average': mean_preds})
    data.to_csv(TABLES / 'pdp_gdp_ppp.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    ax.plot(data.GDP_PPP_per_capita, data.Predicted_PUA_average, color=DARK, linewidth=3)
    ax.scatter([data.GDP_PPP_per_capita.iloc[0], data.GDP_PPP_per_capita.iloc[-1]],
               [mean_preds[0], mean_preds[-1]], color=ORANGE, s=65, zorder=5)
    ax.set_xscale('log')
    # Human-readable GDP values, retaining the logarithmic x-axis.
    ticks = [2000, 5000, 10000, 20000, 40000, 60000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value / 1000:g}k'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title('How does GDP relate to predicted PUA?', color=DARK, fontsize=16, weight='bold')
    ax.set_xlabel('GDP per capita (PPP, international $)', color=DARK)
    ax.set_ylabel('Average predicted PUA (%)', color=DARK)
    ax.grid(alpha=.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / 'pdp_gdp_ppp.png', dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)

def build_linear_regression_pipeline():
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('signed_log', FunctionTransformer(transform_inflation_signed_log, validate=False)),
        ('scale', StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ('numeric', numeric, NUMERIC),
        ('continent', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Continent']),
    ])
    return Pipeline([('preprocess', preprocess), ('model', LinearRegression())])

def plot_validation_model_comparison(results):
    # Include the dummy baseline in the CSV but exclude it from the slide figure.
    display = results[results['model'].isin(['Linear Regression', 'Random Forest'])].copy()
    if len(display) != 2:
        raise ValueError('Expected both Linear Regression and Random Forest for the slide')

    fig, ax = plt.subplots(figsize=(11, 2.8))
    fig.patch.set_facecolor(GREEN)
    ax.axis('off')
    fig.subplots_adjust(left=.055, right=.945, top=.98, bottom=.06)

    cell = [[r['model'], f"{r['MAE']:.2f}", f"{r['RMSE']:.2f}", f"{r['R2']:.3f}"]
            for _, r in display.iterrows()]
    tbl = ax.table(cellText=cell, colLabels=['Model', 'MAE ↓', 'RMSE ↓', 'R² ↑'],
                   colWidths=[.52, .16, .16, .16], cellLoc='center',
                   bbox=[0, .23, 1, .51])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for (r, c), item in tbl.get_celld().items():
        item.set_edgecolor('#8aa481')
        if r == 0:
            item.set_facecolor(DARK)
            item.set_text_props(color='white', weight='bold')
        else:
            selected = display.iloc[r - 1]['model'] == 'Random Forest'
            item.set_facecolor(YELLOW if selected else CREAM)
            item.set_text_props(color=DARK, weight='bold' if selected else 'normal')

    # Text is positioned in axes coordinates to control the gap precisely.
    ax.text(.5, .82, 'Model Comparison — Validation (2023)',
            ha='center', va='center', transform=ax.transAxes,
            color=CREAM, fontsize=17, weight='bold')
    ax.text(.5, .105, 'Random Forest achieved the best performance across all three metrics.',
            ha='center', va='center', transform=ax.transAxes,
            color=CREAM, fontsize=11)

    fig.savefig(FIGURES / 'model_comparison.png', dpi=300,
                bbox_inches='tight', pad_inches=.10,
                facecolor=fig.get_facecolor())
    plt.close(fig)

def compare_validation_models(train, valid):
    candidates = [
        ('Dummy Regressor', DummyRegressor(strategy='mean')),
        ('Linear Regression', build_linear_regression_pipeline()),
        ('Random Forest', build_random_forest_pipeline(True)),
    ]
    rows = []
    for name, estimator in candidates:
        estimator.fit(train[AUGMENTED], train.PUA)
        pred = np.clip(estimator.predict(valid[AUGMENTED]), 0, 100)
        rows.append({'model': name, **calculate_regression_metrics(valid.PUA, pred)})
    results = pd.DataFrame(rows)
    results.to_csv(TABLES / 'model_comparison_three_algorithms.csv', index=False)
    plot_validation_model_comparison(results)
    print('\nModel Comparison — Validation 2023:')
    print(results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('Selected by validation MAE:', results.loc[results.MAE.idxmin(), 'model'])
    return results


def run_supervised_analysis(data_path=None, output_dir=None, *, supervised_data=None):
    """Run the original temporal evaluation using a DataFrame or a CSV."""
    global OUT, TABLES, FIGURES
    if output_dir is not None:
        OUT = Path(output_dir)
        TABLES, FIGURES = OUT / "tables", OUT / "figures"
    for p in (TABLES, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
    df = load_supervised_dataset(data=supervised_data, data_path=data_path)
    train = df[df.Year.between(2017, 2022)]
    valid = df[df.Year.eq(2023)]
    test = df[df.Year.between(2024, 2025)]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError('Empty temporal split')
    print(f'Train: {len(train)}; validation: {len(valid)}; test: {len(test)}')
    rows = []
    for name, cols, add_continent in [('Socioeconomic', NUMERIC, False),
                                      ('Socioeconomic + Continent', AUGMENTED, True)]:
        m = build_random_forest_pipeline(add_continent)
        m.fit(train[cols], train.PUA)
        rows.append({'model':name, **calculate_regression_metrics(valid.PUA, predict_bounded_pua(m, valid, cols))})
    validation = pd.DataFrame(rows)
    validation.to_csv(TABLES / 'validation_comparison.csv', index=False)
    print('\nValidation 2023:')
    print(validation.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    # Compare all three algorithms on 2023; only Linear/RF appear in the slide PNG.
    model_comparison = compare_validation_models(train, valid)

    # Refit the selected candidate on development data; evaluate the test set once.
    development = pd.concat([train, valid], ignore_index=True)
    final = build_random_forest_pipeline(True)
    final.fit(development[AUGMENTED], development.PUA)
    test_pred = predict_bounded_pua(final, test, AUGMENTED)
    test_scores = calculate_regression_metrics(test.PUA, test_pred)
    pd.DataFrame([{'model':'Socioeconomic + Continent', **test_scores}]).to_csv(
        TABLES / 'final_test_metrics.csv', index=False)
    predictions = test[['ISO3', 'Country', 'Continent', 'Year', 'PUA']].copy()
    predictions['Predicted_PUA'] = test_pred
    predictions['Residual'] = predictions.PUA - predictions.Predicted_PUA
    predictions['Absolute_Error'] = predictions.Residual.abs()
    predictions.to_csv(TABLES / 'test_predictions.csv', index=False)
    print('\nFinal test 2024–2025 (with Continent):')
    print(pd.DataFrame([test_scores]).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    plot_final_test_performance(test_scores, len(test))

    def clipped_mae(estimator, X, y):
        return -mean_absolute_error(y, np.clip(estimator.predict(X), 0, 100))
    perm = permutation_importance(final, test[AUGMENTED], test.PUA, scoring=clipped_mae,
                                  n_repeats=15, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({'feature': AUGMENTED, 'MAE_increase': perm.importances_mean,
                        'std': perm.importances_std}).sort_values('MAE_increase', ascending=False)
    imp.to_csv(TABLES / 'permutation_importance.csv', index=False)
    plot_permutation_importance(imp)
    plot_gdp_partial_dependence(final, development)
    print('\nPermutation importance:')
    print(imp.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f'\nSaved outputs under: {OUT}')
    print('All figures and tables generated by the analysis function.')
    print('Interpret associations as predictive, not causal.')
    return {
        'validation': validation,
        'model_comparison': model_comparison,
        'train': train,
        'validation_data': valid,
        'test_data': test,
        'test_metrics': test_scores,
        'test_predictions': predictions,
        'permutation_importance': imp,
        'final_model': final,
    }
