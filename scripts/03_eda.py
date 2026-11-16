from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
from IPython.display import display

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = ROOT / "outputs" / "02_eda"
FIGURES_DIR = OUTPUT_DIR / "figures"
MAPS_DIR = OUTPUT_DIR / "maps"
TABLES_DIR = OUTPUT_DIR / "tables"


NOTEBOOK_DISPLAY = False


def set_notebook_display(enabled=True):
    """Enable or disable inline chart display when functions are called."""
    global NOTEBOOK_DISPLAY
    NOTEBOOK_DISPLAY = bool(enabled)


def ensure_output_directories():
    """Create output folders only when saving results."""
    for directory in (FIGURES_DIR, MAPS_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


BG, DARK = '#FBFBEC', '#07391E'
COLORS = {'Africa':'#E5871E', 'Asia':'#8F6688', 'Europe':'#187D9B',
          'North America':'#B83E65', 'South America':'#659B32', 'Oceania':'#A66B19'}


def load_panel_data(panel=None):
    """Validate an in-memory panel or load the existing processed CSV."""
    options = [ROOT/'data/processed/panel_data.csv', ROOT/'data/processed/panel_original.csv',
               ROOT/'panel_data(2).csv', ROOT/'panel_data.csv']
    if panel is None:
        path = next((p for p in options if p.exists()), None)
        if path is None:
            raise FileNotFoundError('Dataset not found. Paths checked: '+str(options))
        df = pd.read_csv(path)
    else:
        df = panel.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'iso3':'ISO3','country':'Country','year':'Year',
                            'continent':'Continent','pua':'PUA','cohd':'CoHD',
                            'gdp_ppp':'GDP_PPP','inflation':'Inflation',
                            'unemployment':'Unemployment'})
    required = ['ISO3','Year','PUA','CoHD','GDP_PPP','Unemployment','Inflation']
    missing = [x for x in required if x not in df]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if 'Continent' not in df:
        fallback = ROOT/'panel_data(2).csv'
        if fallback.exists():
            lookup = pd.read_csv(fallback, usecols=['ISO3','Continent']).dropna().drop_duplicates('ISO3')
            df = df.merge(lookup, on='ISO3', how='left')
        else:
            raise ValueError('Missing Continent: use a panel with continent labels or panel_data(2).csv')
    for c in ['Year','PUA','CoHD','GDP_PPP','Unemployment','Inflation']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[df.Year.between(2017,2025)].copy()
    df['Continent'] = df['Continent'].replace({'Northern America':'North America',
                                               'Latin America':'South America'})
    return df


def create_chart_axes(title, xlabel='', ylabel='', size=(12,7)):
    fig, ax = plt.subplots(figsize=size, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=19, fontweight='bold', color=DARK, loc='left', pad=22)
    ax.set_xlabel(xlabel, fontsize=16, color=DARK, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=16, color=DARK, labelpad=12)
    ax.tick_params(colors=DARK, labelsize=15)
    ax.grid(alpha=.13, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
    return fig, ax


def save_chart(fig, filename):
    ensure_output_directories()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR/filename, dpi=260, bbox_inches='tight', facecolor=BG)
    if NOTEBOOK_DISPLAY:
        display(fig)
    plt.close(fig)
    print('Saved:', FIGURES_DIR/filename)


def plot_world_choropleth(d, metric, filename, title):
    data = d.dropna(subset=[metric,'ISO3'])
    fig = px.choropleth(data, locations='ISO3', color=metric, hover_name='Country' if 'Country' in data else None,
                        color_continuous_scale=['#E6EEDB','#9DBD80','#E5871E'],
                        range_color=(0,100) if metric=='PUA' else None,
                        labels={metric:'Population unable to afford (%)' if metric=='PUA' else 'PPP $/person/day'},
                        title=title)
    fig.update_geos(projection_type='natural earth', showframe=False, showcountries=True,
                    countrycolor='white', showcoastlines=False, showland=True,
                    landcolor='#E6E8DF', bgcolor=BG)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=DARK,size=17),
                      width=1450, height=800, margin=dict(l=10,r=10,t=90,b=20))
    ensure_output_directories()
    fig.write_html(MAPS_DIR/filename, include_plotlyjs=True)
    if NOTEBOOK_DISPLAY:
        fig.show()
    print('Saved:', MAPS_DIR/filename)


def plot_continent_affordability_boxplots(d):
    d = d.dropna(subset=['PUA','Continent'])
    order = [c for c in COLORS if c in set(d.Continent)]
    vals = [d.loc[d.Continent.eq(c),'PUA'].to_numpy() for c in order]
    fig,ax = create_chart_axes('How unequal is healthy diet affordability? | 2025',
                  'Population unable to afford a healthy diet (%)',size=(12,7))
    box = ax.boxplot(vals, orientation="horizontal", notch=True, showfliers=False, patch_artist=True,
                     widths=.53, positions=range(1,len(order)+1))
    for b,c in zip(box['boxes'],order): b.set(facecolor=COLORS[c],edgecolor=DARK,alpha=.65)
    for median in box['medians']: median.set(color=DARK,linewidth=2.2)
    rng=np.random.default_rng(42)
    for i,(c,v) in enumerate(zip(order,vals),1):
        ax.scatter(v,i+rng.uniform(-.12,.12,len(v)),s=33,alpha=.45,color=COLORS[c],zorder=3)
    ax.set_yticks(range(1,len(order)+1),order)
    ax.invert_yaxis();ax.set_xlim(0,100);ax.grid(axis='x',alpha=.15)
    save_chart(fig,'1.3_pua_distribution_by_continent_2025.png')


def plot_gdp_vs_affordability(d):
    d=d.dropna(subset=['GDP_PPP','PUA','Continent']).query('GDP_PPP > 0').copy()
    if d.empty: return
    mid_x=d.GDP_PPP.median();mid_y=d.PUA.median()
    fig,ax=create_chart_axes('Does wealth guarantee affordability? | 2025',
                'GDP per capita, PPP (international $; log scale)',
                'Population unable to afford a healthy diet (%)',size=(12.5,7.5))
    for c,color in COLORS.items():
        z=d[d.Continent.eq(c)]
        ax.scatter(z.GDP_PPP,z.PUA,s=115,alpha=.78,color=color,label=c,edgecolor=BG,linewidth=.45,zorder=3)
    ax.set_xscale('log');ax.set_ylim(0,100)
    ax.axvline(mid_x,color=DARK,linestyle='--',alpha=.45)
    ax.axhline(mid_y,color=DARK,linestyle='--',alpha=.45)
    ax.text(.02,.97,'LOW INCOME · HIGH PUA',transform=ax.transAxes,va='top',color=DARK,fontsize=13,alpha=.8)
    ax.text(.98,.97,'HIGH INCOME · HIGH PUA',transform=ax.transAxes,ha='right',va='top',color=DARK,fontsize=13,alpha=.8)
    ax.text(.02,.03,'LOW INCOME · LOW PUA',transform=ax.transAxes,color=DARK,fontsize=13,alpha=.8)
    ax.text(.98,.03,'HIGH INCOME · LOW PUA',transform=ax.transAxes,ha='right',color=DARK,fontsize=13,alpha=.8)
    ax.legend(frameon=False,title='Continent',bbox_to_anchor=(1.02,1),loc='upper left', fontsize=14, title_fontsize=15)
    save_chart(fig,'2.1_wealth_vs_affordability_quadrants_2025.png')


def plot_unemployment_affordability_gap(d):
    """Compare median PUA across low- and high-unemployment countries.

    The threshold is the global median unemployment rate among countries
    with both variables available in 2025. Each country has equal weight.
    """
    d = d.dropna(subset=['Unemployment', 'PUA', 'Continent']).copy()
    d = d[d['Unemployment'].ge(0) & d['PUA'].between(0, 100)]
    if d.empty:
        print('2.2: no data available; chart skipped.')
        return
    threshold = d['Unemployment'].median()
    d['group'] = np.where(d['Unemployment'] < threshold, 'Low unemployment',
                          'High unemployment')
    summary = (d.groupby(['Continent', 'group'], observed=True)
               .agg(median_pua=('PUA', 'median'), n=('PUA', 'size'))
               .reset_index())
    order = [c for c in COLORS if c in set(d.Continent)]
    low_color, high_color = '#187D9B', '#E5871E'
    fig, ax = create_chart_axes('The Unemployment Divide | 2025',
                   'Median population unable to afford a healthy diet (%)',
                   size=(14.5, 7.4))
    plotted = 0
    for i, continent in enumerate(order):
        z = summary[summary.Continent.eq(continent)].set_index('group')
        if not {'Low unemployment', 'High unemployment'}.issubset(z.index):
            continue
        low = z.loc['Low unemployment']
        high = z.loc['High unemployment']
        xl, xh = float(low.median_pua), float(high.median_pua)
        ax.plot([xl, xh], [i, i], color='#A9B7A9', lw=3.4,
                solid_capstyle='round', zorder=2)
        ax.scatter([xl], [i], s=230, color=low_color, edgecolor=BG,
                   linewidth=1.6, zorder=4)
        ax.scatter([xh], [i], s=230, color=high_color, edgecolor=BG,
                   linewidth=1.6, zorder=5)
        # Numeric labels offset above/below to avoid collisions when medians are close.
        ax.annotate(f'{xl:.1f}%  (n={int(low.n)})', (xl, i),
                    xytext=(0, -16), textcoords='offset points',
                    ha='center', va='top', fontsize=12, color=low_color,
                    fontweight='bold')
        ax.annotate(f'{xh:.1f}%  (n={int(high.n)})', (xh, i),
                    xytext=(0, 16), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12, color=high_color,
                    fontweight='bold')
        plotted += 1
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlim(-3, 105)
    ax.set_ylim(len(order)-.45, -.55)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis='x', alpha=.18)
    ax.grid(axis='y', visible=False)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=12,
                      markerfacecolor=low_color, markeredgecolor=BG,
                      label='Low unemployment'),
               Line2D([0], [0], marker='o', linestyle='none', markersize=12,
                      markerfacecolor=high_color, markeredgecolor=BG,
                      label='High unemployment')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              ncol=1, frameon=False, fontsize=14, borderaxespad=0)
    save_chart(fig, '2.2_unemployment_divide_2025.png')
    print(f'2.2: {plotted} continents with both groups; threshold = {threshold:.2f}%.')


def plot_inflation_affordability_heatmap(df):
    """Heatmap 2018–2025: color = median inflation by continent/year;
    labels = median PUA change (percentage points) for countries observed in both years.
    """
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from matplotlib.cm import ScalarMappable

    data = df[['ISO3', 'Continent', 'Year', 'Inflation', 'PUA']].copy()
    data = data.dropna(subset=['ISO3', 'Continent', 'Year'])
    data = data.sort_values(['ISO3', 'Year']).drop_duplicates(['ISO3', 'Year'])
    previous = data[['ISO3', 'Year', 'PUA']].copy()
    previous['Year'] += 1
    previous = previous.rename(columns={'PUA': 'PUA_previous'})
    data = data.merge(previous, on=['ISO3', 'Year'], how='left')
    data = data[data.Year.between(2018, 2025)].copy()
    data['pua_change_pp'] = data['PUA'] - data['PUA_previous']
    # Include a country only if inflation and PUA are available in both years.
    data = data.dropna(subset=['Inflation', 'pua_change_pp'])
    summary = (data.groupby(['Continent', 'Year'], as_index=False)
               .agg(inflation=('Inflation', 'median'),
                    pua_change=('pua_change_pp', 'median'),
                    n=('ISO3', 'nunique')))
    order = [c for c in COLORS if c in set(summary.Continent)]
    years = list(range(2018, 2026))
    if not order:
        print('2.3: no data available for the heatmap.')
        return
    infl = summary.pivot(index='Continent', columns='Year', values='inflation').reindex(index=order, columns=years)
    changes = summary.pivot(index='Continent', columns='Year', values='pua_change').reindex(index=order, columns=years)

    fig, ax = plt.subplots(figsize=(14.5, 7.2), facecolor=BG)
    ax.set_facecolor(BG)
    vals = infl.to_numpy(dtype=float)
    valid = vals[np.isfinite(vals)]
    # Robust limits affect only the color scale; no observations are excluded.
    lo, hi = np.percentile(valid, [5, 95])
    if hi <= lo: hi = lo + 1
    norm = Normalize(vmin=lo, vmax=hi, clip=True)
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad('#E6E8DF')
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(np.arange(len(years)), [str(y) for y in years], fontsize=15, color=DARK)
    ax.set_yticks(np.arange(len(order)), order, fontsize=15, color=DARK)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which='minor', color=BG, linewidth=5)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values(): spine.set_visible(False)
    for i, continent in enumerate(order):
        for j, year in enumerate(years):
            delta = changes.loc[continent, year]
            if pd.notna(delta):
                ink = 'white' if norm(vals[i,j]) > .68 else DARK
                ax.text(j, i, f'{delta:+.1f}', ha='center', va='center',
                        fontsize=15, fontweight='bold', color=ink)
    ax.set_title('Inflation & Affordability | 2018–2025', loc='left',
                 fontsize=20, fontweight='bold', color=DARK, pad=23)
    ax.set_xlabel('Year', fontsize=16, color=DARK, labelpad=12)
    ax.set_ylabel('Continent', fontsize=16, color=DARK, labelpad=12)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    cb.set_label('Median annual inflation (%)', color=DARK, fontsize=15, labelpad=10)
    cb.ax.tick_params(colors=DARK, labelsize=14)
    cb.outline.set_visible(False)
    # Cell labels show PUA changes in percentage points.
    save_chart(fig, '2.3_inflation_affordability_heatmap_2018_2025.png')




def plot_continent_time_trends(df, metric, title, filename):
    """Plot the mean of country-level values by continent, 2017–2025."""
    d = df.dropna(subset=['Continent', 'Year', metric]).copy()
    trends = d.groupby(['Continent', 'Year'], as_index=False)[metric].mean()
    fig, ax = create_chart_axes(title, 'Year',
                   'Population unable to afford a healthy diet (%)' if metric == 'PUA'
                   else 'Cost of a healthy diet (PPP $/person/day)', size=(14.5, 7.5))
    # Show country-level series as thin, transparent background lines.
    # These lines do not change continental means or source data.
    countries = d.dropna(subset=['ISO3']).sort_values('Year')
    for (continent, iso3), country in countries.groupby(['Continent', 'ISO3']):
        if len(country) >= 2:
            ax.plot(country['Year'], country[metric],
                    color=COLORS.get(continent, '#AAB4A5'),
                    linewidth=0.75, alpha=0.075, zorder=1,
                    solid_capstyle='round')

    for continent, color in COLORS.items():
        z = trends[trends.Continent.eq(continent)].sort_values('Year')
        if not z.empty:
            ax.plot(z.Year, z[metric], color=color, lw=3.7, marker='o',
                    markersize=9, markeredgecolor=BG, markeredgewidth=1.2,
                    label=continent, zorder=4)
    ax.set_xticks(range(2017, 2026))
    ax.set_xlim(2016.8, 2025.2)
    if metric == 'PUA': ax.set_ylim(0, 100)
    else: ax.set_ylim(bottom=0)
    ax.grid(alpha=.13)
    ax.legend(title='Continent', loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=False, fontsize=15, title_fontsize=16)
    save_chart(fig, filename)


def plot_cost_vs_affordability_changes(df):
    """Plot continental averages of country-level changes between 2017 and 2025."""
    d = df.dropna(subset=['ISO3', 'Year']).drop_duplicates(['ISO3', 'Year'])
    start = d[d.Year.eq(2017)].set_index('ISO3')[['CoHD', 'PUA']]
    end = d[d.Year.eq(2025)].set_index('ISO3')[['CoHD', 'PUA', 'Continent']]
    both = start.join(end, how='inner', lsuffix='_2017', rsuffix='_2025')
    both = both.dropna(subset=['CoHD_2017', 'CoHD_2025', 'PUA_2017',
                                'PUA_2025', 'Continent'])
    both = both[both.CoHD_2017.gt(0)].copy()
    if both.empty:
        print('3.3: no country has complete data for both 2017 and 2025.')
        return
    both['cost_change_pct'] = 100 * (both.CoHD_2025 / both.CoHD_2017 - 1)
    both['pua_change_pp'] = both.PUA_2025 - both.PUA_2017
    summary = both.groupby('Continent', as_index=False).agg(
        cost_change=('cost_change_pct', 'mean'),
        pua_change=('pua_change_pp', 'mean'), countries=('cost_change_pct', 'size'))
    order = [c for c in COLORS if c in set(summary.Continent)]
    summary = summary.set_index('Continent').loc[order]
    ensure_output_directories()
    summary.to_csv(TABLES_DIR / '3.3_continent_mean_changes.csv')
    y = np.arange(len(order))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7), sharey=True,
                               facecolor=BG, gridspec_kw={'wspace': .25})
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.axvline(0, color=DARK, lw=1.4, alpha=.5)
        ax.grid(axis='x', alpha=.13)
        ax.set_axisbelow(True)
        ax.tick_params(colors=DARK, labelsize=15, length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    cost = summary.cost_change.to_numpy()
    pua = summary.pua_change.to_numpy()
    ax1.barh(y, cost, height=.58, color=[COLORS[c] for c in order], zorder=3)
    ax2.barh(y, pua, height=.58,
             color=['#659B32' if v < 0 else '#B83E65' if v > 0 else '#A9B7A9' for v in pua],
             zorder=3)
    ax1.set_yticks(y, order)
    ax1.invert_yaxis()
    ax2.tick_params(axis='y', left=False, labelleft=False)
    ax1.set_title('Healthy diet cost', fontsize=19, fontweight='bold', color=DARK, pad=20)
    ax2.set_title('Population unable to afford', fontsize=19, fontweight='bold', color=DARK, pad=20)
    ax1.set_xlabel('Mean CoHD change (%)', fontsize=16, color=DARK, labelpad=13)
    ax2.set_xlabel('Mean PUA change (percentage points)', fontsize=16, color=DARK, labelpad=13)
    for ax, values, suffix in [(ax1, cost, '%'), (ax2, pua, ' pp')]:
        low, high = min(0, values.min()), max(0, values.max())
        margin = max((high - low) * .16, 2)
        ax.set_xlim(low - margin * 1.8, high + margin * 1.8)
        for i, v in enumerate(values):
            ax.annotate(f'{v:+.1f}{suffix}', (v, i),
                        xytext=(9 if v >= 0 else -9, 0), textcoords='offset points',
                        ha='left' if v >= 0 else 'right', va='center',
                        fontsize=14, fontweight='bold', color=DARK)
    fig.suptitle('Rising Costs vs. Affordability | 2017–2025',
                 x=.08, ha='left', fontsize=21, fontweight='bold', color=DARK)
    fig.subplots_adjust(top=.83, bottom=.13, left=.16, right=.96)
    ensure_output_directories()
    fig.savefig(FIGURES_DIR / '3.3_cohd_change_vs_pua_change.png', dpi=260,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('Saved:', FIGURES_DIR / '3.3_cohd_change_vs_pua_change.png')



# -----------------------------------------------------------------------------
# RQ4 — What's Behind the Price of a Healthy Diet? (2021)
# -----------------------------------------------------------------------------
FOOD_COLORS = {
    "Animal-source foods": "#C95618",
    "Vegetables": "#8FA653",
    "Fruits": "#E8A42A",
    "Starchy staples": "#97704B",
    "Legumes, nuts and seeds": "#D7A34B",
    "Oils and fats": "#B19A0B",
}

# Dataset columns mapped to chart labels and the existing food-group palette.
FOOD_GROUPS = {
    "Animal_Foods": ("Animal-source foods", FOOD_COLORS["Animal-source foods"]),
    "Vegetables": ("Vegetables", FOOD_COLORS["Vegetables"]),
    "Fruits": ("Fruits", FOOD_COLORS["Fruits"]),
    "Starchy_Staples": ("Starchy staples", FOOD_COLORS["Starchy staples"]),
    "Legumes_Nuts_Seeds": ("Legumes, nuts and seeds", FOOD_COLORS["Legumes, nuts and seeds"]),
    "Oils_Fats": ("Oils and fats", FOOD_COLORS["Oils and fats"]),
}


def load_food_group_costs_2021(panel, food_data=None):
    options = [ROOT / 'data/processed/food_categories_2021.csv',
               ROOT / 'food_categories_2021.csv']
    if food_data is None:
        path = next((p for p in options if p.exists()), None)
        if path is None:
            raise FileNotFoundError('Missing data/processed/food_categories_2021.csv')
        food = pd.read_csv(path, encoding='utf-8-sig')
    else:
        food = food_data.copy()
    food.columns = food.columns.str.strip()
    missing = [c for c in ['ISO3', *FOOD_GROUPS] if c not in food.columns]
    if missing:
        raise ValueError(f'Missing columns in food dataset: {missing}')
    mapping = panel.loc[panel.Year.eq(2021), ['ISO3', 'Continent']].dropna().drop_duplicates('ISO3')
    if 'Continent' in food.columns:
        food = food.drop(columns=['Continent'])
    food = food.merge(mapping, on='ISO3', how='inner', validate='many_to_one')
    for col in FOOD_GROUPS:
        food[col] = pd.to_numeric(food[col], errors='coerce')
    return food


def plot_food_group_costs(food):
    # Use the same set of complete countries for charts 4.1 and 4.2.
    complete = food.dropna(subset=list(FOOD_GROUPS)).copy()
    complete = complete[(complete[list(FOOD_GROUPS)] >= 0).all(axis=1)]
    means = complete[list(FOOD_GROUPS)].mean().sort_values(ascending=False)
    fig, ax = create_chart_axes('What’s on the Bill? | 2021',
                   'Mean cost (PPP $ per person per day)', size=(14, 8))
    y = np.arange(len(means))
    xmax = means.max() * 1.28
    ax.set_xlim(0, xmax)
    for i, (col, val) in enumerate(means.items()):
        color = FOOD_GROUPS[col][1]
        ax.plot([0, val], [i, i], lw=9, alpha=.48, color=color,
                solid_capstyle='round', zorder=2)
        ax.scatter(val, i, s=480, color=color, edgecolor=BG, linewidth=2.5, zorder=4)
        ax.text(val + xmax * .035, i, f'{val:.2f}', va='center',
                fontsize=17, color=DARK, fontweight='bold')
    ax.set_yticks(y, [FOOD_GROUPS[c][0] for c in means.index])
    ax.tick_params(axis='y', length=0, pad=12, labelsize=16)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.13)
    ax.grid(axis='y', visible=False)
    save_chart(fig, '4.1_food_category_costs_2021.png')
    print('4.1: complete countries:', len(complete))


def plot_continent_food_cost_shares(food):
    complete = food.dropna(subset=list(FOOD_GROUPS)).copy()
    complete = complete[(complete[list(FOOD_GROUPS)] >= 0).all(axis=1)]
    complete = complete[complete[list(FOOD_GROUPS)].sum(axis=1).gt(0)]
    shares = complete[list(FOOD_GROUPS)].div(complete[list(FOOD_GROUPS)].sum(axis=1), axis=0) * 100
    shares['Continent'] = complete['Continent'].to_numpy()
    summary = shares.groupby('Continent')[list(FOOD_GROUPS)].mean()
    order = [c for c in COLORS if c in summary.index]
    summary = summary.loc[order]
    fig, ax = create_chart_axes('Different Continents, Different Plates | 2021',
                   'Mean share of healthy diet cost (%)', size=(15, 8))
    left = np.zeros(len(order))
    y = np.arange(len(order))
    for col, (label, color) in FOOD_GROUPS.items():
        vals = summary[col].to_numpy()
        ax.barh(y, vals, left=left, color=color, height=.63,
                label=label, edgecolor=BG, linewidth=1.1, zorder=3)
        left += vals
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.tick_params(axis='y', length=0, labelsize=16)
    ax.legend(title='Food group', bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=False, fontsize=15, title_fontsize=16)
    save_chart(fig, '4.2_food_cost_composition_by_continent_2021.png')
    print('4.2: complete countries:', len(complete))


def plot_diet_cost_vs_unaffordability(panel):
    d = panel[panel.Year.eq(2021)].dropna(subset=['CoHD', 'PUA', 'Continent']).copy()
    fig, ax = create_chart_axes('Expensive Doesn’t Always Mean Unaffordable | 2021',
                   'Cost of a healthy diet (PPP $ per person per day)',
                   'Population unable to afford a healthy diet (%)', size=(13.5, 8))
    for continent, color in COLORS.items():
        sub = d[d.Continent.eq(continent)]
        if not sub.empty:
            ax.scatter(sub.CoHD, sub.PUA, s=115, alpha=.75, color=color,
                       edgecolors=BG, linewidths=.8, label=continent, zorder=3)
    ax.set_ylim(0, 100)
    ax.legend(title='Continent', bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=False, fontsize=15, title_fontsize=16)
    save_chart(fig, '4.3_cohd_vs_pua_2021.png')


def create_food_cost_charts(panel, food_data=None):
    food = load_food_group_costs_2021(panel, food_data=food_data)
    plot_food_group_costs(food)
    plot_continent_food_cost_shares(food)
    plot_diet_cost_vs_unaffordability(panel)
