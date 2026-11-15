from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_project_root() -> Path:
    """Locate the project root from the scripts directory."""
    return Path(__file__).resolve().parent.parent


ROOT = find_project_root()
ML = ROOT / "data" / "processed" / "ml"
OUT = ROOT / "outputs" / "04_unsupervised"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

def ensure_output_directories() -> None:
    """Create output directories only when the analysis is executed."""
    for directory in (TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

ECON_FEATURES = ["log_GDP_PPP", "Urban", "Unemployment", "Inflation",]
META = ["ISO3", "Country", "Continent",]
TARGETS = ["CoHD", "PUA",]

RANDOM_STATE = 42
K_VALUES = range(2, 9)

PCA_PAIRS = [(1, 2), (2, 3), (1, 3),]

CLUSTER_COLORS = ["#187D9B", "#8F6688", "#E5871E", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]  # Stable colors for the first three clusters.

def load_data(filename, required):
    """Load a dataset and validate the required columns."""
    path = ML / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. "
            "Run 04_preprocessing.py first."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"{filename}: missing columns {missing}"
        )
    if df["ISO3"].duplicated().any():
        raise ValueError(
            f"{filename}: duplicate ISO3 codes. "
            "Expected one row per country."
        )
    return df

def prepare_features(df, features):
    """
    Prepare the socioeconomic features.
    Inflation is transformed using signed log
    to reduce the impact of extreme values.
    """
    x = (
        df[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    valid = x.notna().all(axis=1)
    if (~valid).any():
        print(
            f"Excluded {int((~valid).sum())} rows "
            "with missing features."
        )
    data = df.loc[valid].copy().reset_index(drop=True)
    x = x.loc[valid].reset_index(drop=True)
    if len(data) < 10:
        raise ValueError(
            "Too few complete observations "
            "to run clustering."
        )
    # Apply the signed-log transformation to inflation.
    x_transformed = x.copy()
    x_transformed["Inflation"] = (
        np.sign(x["Inflation"])
        * np.log1p(np.abs(x["Inflation"]))
    )
    # Standardize the clustering features.
    scaler = StandardScaler()
    z = scaler.fit_transform(x_transformed)
    # Save transformed features for diagnostics.
    transformed = data[META].copy()
    for feature in features:
        transformed[feature] = x_transformed[feature]
    transformed.to_csv(
        TABLES / "economic_transformed_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # Save standardized features for diagnostics.
    standardized = data[META].copy()
    for i, feature in enumerate(features):
        standardized[feature] = z[:, i]
    standardized.to_csv(
        TABLES / "economic_standardized_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return data, x, z

def pca_analysis(df, z, features, prefix):
    """Fit PCA and report explained variance."""
    pca = PCA()
    scores = pca.fit_transform(z)
    variance = pd.DataFrame({
        "PC": [
            f"PC{i + 1}"
            for i in range(len(features))
        ],
        "explained_variance_ratio": (
            pca.explained_variance_ratio_
        ),
        "cumulative_variance": np.cumsum(
            pca.explained_variance_ratio_
        ),
    })
    variance.to_csv(
        TABLES / f"{prefix}_pca_variance.csv",
        index=False,
    )
    print("\n=== PCA EXPLAINED VARIANCE ===")
    for i, ratio in enumerate(
        pca.explained_variance_ratio_
    ):
        cumulative = variance.iloc[i][
            "cumulative_variance"
        ]

        print(
            f"PC{i + 1}: {ratio * 100:.2f}% "
            f"| Cumulative: {cumulative * 100:.2f}%"
        )
    print(
        "\nPC1 + PC2: "
        f"{variance.iloc[1]['cumulative_variance'] * 100:.2f}%"
    )
    # Save PCA loadings.
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[
            f"PC{i + 1}"
            for i in range(len(features))
        ],
    )
    loadings.index = [
        "Signed log Inflation"
        if feature == "Inflation"
        else feature
        for feature in loadings.index
    ]
    loadings.to_csv(
        TABLES / f"{prefix}_pca_loadings.csv",
        index_label="feature",
    )
    # Save PCA scores.
    coordinates = df[META].copy()
    for i in range(len(features)):
        coordinates[f"PC{i + 1}"] = scores[:, i]
    coordinates.to_csv(
        TABLES / f"{prefix}_pca_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # Plot explained and cumulative variance.
    fig, ax = plt.subplots(figsize=(7, 4))
    components = range(1, len(features) + 1)
    ax.plot(
        components,
        pca.explained_variance_ratio_,
        "o-",
        label="Explained variance",
    )
    ax.plot(
        components,
        np.cumsum(pca.explained_variance_ratio_),
        "s--",
        label="Cumulative variance",
    )
    ax.set(
        xlabel="Principal component",
        ylabel="Proportion of variance explained",
        title="Socioeconomic PCA - Scree plot",
    )
    ax.set_xticks(list(components))
    ax.legend()
    fig.patch.set_facecolor("#FBFBEC")
    ax.set_facecolor("#FBFBEC")
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{prefix}_scree_plot.png",
        dpi=180,
    )
    plt.close(fig)
    return coordinates, pca.explained_variance_ratio_

def select_k(z, prefix):
    """Select K using the highest silhouette score."""
    rows = []
    max_k = min(max(K_VALUES), len(z) - 1)
    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )
        labels = model.fit_predict(z)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(
                z, labels
            ),
        })
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(
        TABLES / f"{prefix}_k_selection.csv",
        index=False,
    )
    best_k = int(
        evaluation.loc[
            evaluation["silhouette"].idxmax(),
            "k",
        ]
    )
    print("\n=== K SELECTION ===")
    print(evaluation.to_string(index=False))
    print(f"\nSelected K: {best_k}")

    # Plot silhouette scores by cluster count.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        evaluation["k"],
        evaluation["silhouette"],
        "o-",
    )
    ax.axvline(
        best_k,
        linestyle="--",
        color="black",
        linewidth=1,
        label=f"Selected K = {best_k}",
    )
    ax.set(
        xlabel="Number of clusters (K)",
        ylabel="Silhouette score",
        title="Selecting the number of clusters - Silhouette",
    )
    ax.legend()
    fig.patch.set_facecolor("#FBFBEC")
    ax.set_facecolor("#FBFBEC")
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{prefix}_silhouette.png",
        dpi=180,
    )
    plt.close(fig)

    # Plot inertia by cluster count.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        evaluation["k"],
        evaluation["inertia"],
        "o-",
    )
    ax.set(
        xlabel="Number of clusters (K)",
        ylabel="Inertia",
        title="Selecting the number of clusters - Elbow method",
    )
    fig.patch.set_facecolor("#FBFBEC")
    ax.set_facecolor("#FBFBEC")
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{prefix}_elbow.png",
        dpi=180,
    )
    plt.close(fig)
    return best_k

def plot_pca_clusters(
    coordinates,
    labels,
    explained_variance,
    prefix,
):
    print("\n=== PCA SCATTERPLOTS ===")
    for pc_x, pc_y in PCA_PAIRS:
        fig, ax = plt.subplots(figsize=(9, 6))
        for cluster in sorted(np.unique(labels)):
            idx = labels == cluster
            ax.scatter(
                coordinates.loc[idx, f"PC{pc_x}"],
                coordinates.loc[idx, f"PC{pc_y}"],
                label=f"Cluster {cluster + 1}",
                color=CLUSTER_COLORS[cluster],
                alpha=0.75,
                s=35,
                edgecolors="white",
                linewidths=0.3,
            )
        variance_x = (
            explained_variance[pc_x - 1] * 100
        )
        variance_y = (
            explained_variance[pc_y - 1] * 100
        )
        total_variance = variance_x + variance_y
        ax.set(
            xlabel=(
                f"PC{pc_x} "
                f"({variance_x:.1f}%)"
            ),
            ylabel=(
                f"PC{pc_y} "
                f"({variance_y:.1f}%)"
            ),
            title=(
                f"K-Means clusters - PC{pc_x} vs PC{pc_y}"
                f"\nExplained variance: "
                f"{total_variance:.1f}%"
            ),
        )
        ax.axhline(
            0,
            color="gray",
            linewidth=0.5,
            alpha=0.5,
        )
        ax.axvline(
            0,
            color="gray",
            linewidth=0.5,
            alpha=0.5,
        )
        ax.legend(
            title="Socioeconomic cluster",
            loc="best",
        )
        ax.grid(
            alpha=0.15,
            linestyle="--",
        )
        fig.patch.set_facecolor("#FBFBEC")
        ax.set_facecolor("#FBFBEC")
        fig.tight_layout()
        filename = (
            f"{prefix}_pca_clusters_"
            f"pc{pc_x}_pc{pc_y}.png"
        )
        fig.savefig(
            FIGURES / filename,
            dpi=180,
        )
        plt.close(fig)
        print(
            f"PC{pc_x}-PC{pc_y}: "
            f"{total_variance:.2f}% "
            "explained variance"
        )
    print(
        f"\nSaved {len(PCA_PAIRS)} PCA scatterplots."
    )

def run_clustering(
    df,
    x,
    z,
    coordinates,
    explained_variance,
    prefix,
):
    """Fit K-Means and summarize cluster assignments."""
    best_k = select_k(z, prefix)
    model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    labels = model.fit_predict(z)
    # Save cluster assignments.
    assignments = df[META + TARGETS].copy()
    assignments["Cluster"] = labels + 1
    assignments.to_csv(
        TABLES / f"{prefix}_cluster_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # Summarize cluster profiles in the original feature scale.
    profiles = x.copy()
    profiles["Cluster"] = labels + 1
    for target in TARGETS:
        profiles[target] = df[target].values

    summary = profiles.groupby("Cluster").agg(
        ["mean", "median", "count"]
    )
    summary.to_csv(
        TABLES / f"{prefix}_cluster_profiles.csv"
    )
    # Save the number of countries in each cluster.
    counts = (
        assignments.groupby("Cluster")
        .size()
        .reset_index(name="countries")
    )
    counts.to_csv(
        TABLES / f"{prefix}_cluster_sizes.csv",
        index=False,
    )
    # Plot the three selected PCA projections.
    plot_pca_clusters(
        coordinates,
        labels,
        explained_variance,
        prefix,
    )
    silhouette = silhouette_score(z, labels)

    print("\n=== K-MEANS RESULTS ===")
    print(f"Countries analyzed: {len(df)}")
    print(f"Number of clusters: {best_k}")
    print(f"Silhouette Score: {silhouette:.3f}")
    print("\nCluster sizes:")
    print(counts.to_string(index=False))

    return assignments

def cluster_heatmap(
    z,
    features,
    assignments,
    prefix,
):
    """Plot standardized cluster profiles as a heatmap."""
    standardized = pd.DataFrame(
        z,
        columns=features,
    )
    standardized["Cluster"] = (
        assignments["Cluster"].values
    )
    cluster_means = (
        standardized
        .groupby("Cluster")[features]
        .mean()
    )
    cluster_means.to_csv(
        TABLES
        / f"{prefix}_cluster_standardized_profiles.csv"
    )
    labels = [
        "Log GDP PPP",
        "Urbanization",
        "Unemployment",
        "Signed log Inflation",
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    limit = np.abs(
        cluster_means.to_numpy()
    ).max()
    im = ax.imshow(
        cluster_means,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(cluster_means)))
    ax.set_yticklabels([
        f"Cluster {k}"
        for k in cluster_means.index
    ])
    for i in range(len(cluster_means)):
        for j in range(len(features)):
            value = cluster_means.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if abs(value) > limit * 0.55
                    else "black"
                ),
                fontsize=11,
            )
    ax.set_title(
        "Socioeconomic cluster profiles - Standardized means"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Mean standardized value (z-score)"
    )
    fig.patch.set_facecolor("#FBFBEC")
    ax.set_facecolor("#FBFBEC")
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{prefix}_cluster_heatmap.png",
        dpi=180,
    )
    plt.close(fig)

def world_cluster_map(assignments, prefix, all_countries=None):
    """Export a world cluster map using the full country dataset."""
    if all_countries is None:
        all_countries = load_data(
            "unsupervised_2021_all.csv",
            META,
        )
    missing = sorted(set(META) - set(all_countries.columns))
    if missing:
        raise ValueError(f"Missing map columns: {missing}")
    map_data = all_countries[META].merge(
        assignments[
            ["ISO3", "Cluster", "CoHD", "PUA"]
        ],
        on="ISO3",
        how="left",
        validate="one_to_one",
    )
    map_data["Cluster_Label"] = (
        map_data["Cluster"].apply(
            lambda value: (
                f"Cluster {int(value)}"
                if pd.notna(value)
                else "Not clustered"
            )
        )
    )
    # Use the same cluster colors as the PCA scatterplots.
    color_map = {
        f"Cluster {i + 1}": color
        for i, color in enumerate(CLUSTER_COLORS)
    }
    color_map["Not clustered"] = "#D9D9D9"
    cluster_order = [
        f"Cluster {int(k)}"
        for k in sorted(
            assignments["Cluster"].unique()
        )
    ]
    cluster_order.append("Not clustered")
    # Create the world choropleth.
    fig = px.choropleth(
        map_data,
        locations="ISO3",
        locationmode="ISO-3",
        color="Cluster_Label",
        color_discrete_map=color_map,
        category_orders={
            "Cluster_Label": cluster_order
        },
        title="World map of socioeconomic clusters (2021)",
    )
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.4,
    )
    fig.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=False,
        showcountries=True,
        countrycolor="white",
        showland=True,
        landcolor="#EEEEEE",
        bgcolor="#FBFBEC",
    )
    fig.update_layout(
        title_x=0.5,
        template="plotly_white",
        paper_bgcolor="#FBFBEC",
        legend_title_text="Socioeconomic cluster",
        margin=dict(
            l=10,
            r=10,
            t=65,
            b=10,
        ),
        width=1600,
        height=850,
    )
    fig.write_image(
        FIGURES
        / f"{prefix}_world_cluster_map.png",
        width=1600,
        height=850,
        scale=2,
    )
    # Save the data underlying the map.
    map_data.to_csv(
        TABLES
        / f"{prefix}_world_cluster_map_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n=== WORLD MAP ===")
    print(
        "Countries with clusters:",
        map_data["Cluster"].notna().sum(),
    )
    print(
        "Countries without clusters:",
        map_data["Cluster"].isna().sum(),
    )

def run_unsupervised_analysis(unsupervised_data=None, all_countries=None):
    """
    Run PCA and K-Means and interpret the clusters
    using CoHD and PUA.
    """
    ensure_output_directories()
    print("\n=== 2021 SOCIOECONOMIC CLUSTERING ===")
    if unsupervised_data is None:
        df = load_data(
            "unsupervised_2021.csv",
            META + ECON_FEATURES + TARGETS,
        )
    else:
        df = unsupervised_data.copy()
        missing = sorted(set(META + ECON_FEATURES + TARGETS) - set(df.columns))
        if missing:
            raise ValueError(f"Missing clustering columns: {missing}")
        if df["ISO3"].duplicated().any():
            raise ValueError("Expected one row per country (unique ISO3).")
    # Prepare clustering features.
    data, x, z = prepare_features(
        df,
        ECON_FEATURES,
    )
    # Run PCA.
    coordinates, explained_variance = pca_analysis(
        data,
        z,
        ECON_FEATURES,
        "economic",
    )
    # Run K-Means.
    assignments = run_clustering(
        data,
        x,
        z,
        coordinates,
        explained_variance,
        "economic",
    )
    # Plot standardized cluster profiles.
    cluster_heatmap(
        z,
        ECON_FEATURES,
        assignments,
        "economic",
    )
    # Export the world map as a PNG.
    world_cluster_map(
        assignments,
        "economic",
        all_countries=all_countries,
    )
    # Compare CoHD and PUA after clustering.
    for target in TARGETS:

        comparison = (
            assignments
            .groupby("Cluster")[target]
            .agg([
                "count",
                "mean",
                "median",
                "std",
            ])
        )
        comparison.to_csv(
            TABLES
            / f"economic_{target.lower()}_by_cluster.csv"
        )
        print(f"\n=== {target} BY CLUSTER ===")
        print(comparison.round(2).to_string())
        clusters = sorted(
            assignments["Cluster"].unique()
        )
        groups = [
            assignments.loc[
                assignments["Cluster"].eq(k),
                target,
            ].dropna().values
            for k in clusters
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        boxes = ax.boxplot(
            groups,
            notch=True,
            showmeans=True,
            patch_artist=True,
            tick_labels=[
                f"Cluster {k}"
                for k in clusters
            ],
        )
        for box, cluster in zip(boxes["boxes"], clusters):
            box.set_facecolor(CLUSTER_COLORS[int(cluster) - 1])
            box.set_alpha(0.75)
        if target == "CoHD":
            ylabel = (
                "Cost of a healthy diet "
                "(PPP $/person/day)"
            )
            title = (
                "Cost of a healthy diet "
                "by socioeconomic cluster"
            )
        else:
            ylabel = (
                "Population unable to afford "
                "a healthy diet (%)"
            )
            title = (
                "Healthy diet unaffordability "
                "by socioeconomic cluster"
            )
        ax.set(
            xlabel="Socioeconomic cluster",
            ylabel=ylabel,
            title=title,
        )
        fig.patch.set_facecolor("#FBFBEC")
        ax.set_facecolor("#FBFBEC")
        fig.tight_layout()
        fig.savefig(
            FIGURES
            / f"economic_{target.lower()}_by_cluster.png",
            dpi=180,
        )

        plt.close(fig)

    return {"data": data, "pca_scores": coordinates, "cluster_assignments": assignments}


