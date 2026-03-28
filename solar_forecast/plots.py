"""
plots.py
Reusable plot functions for EDA and error analysis.
Importable by notebooks and by reporting scripts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_capacity_by_zone(capacity_plot_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(data=capacity_plot_df, x="zone_name", y="capacity_nonmissing", ax=axes[0])
    axes[0].set_title("Capacity Coverage by Zone")
    axes[0].set_xlabel("Zone")
    axes[0].set_ylabel("Non-Missing Capacity Rows")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=capacity_plot_df, x="zone_name", y="capacity_max", ax=axes[1])
    axes[1].set_title("Maximum Capacity by Zone")
    axes[1].set_xlabel("Zone")
    axes[1].set_ylabel("Capacity MW")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_zone_summary(zone_summary: pd.DataFrame) -> None:
    zone_error_plot = zone_summary.sort_values("mae", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.barplot(data=zone_summary, x="zone_name", y="actual_mean", ax=axes[0])
    axes[0].set_title("Average Actual Solar Generation by Zone")
    axes[0].set_xlabel("Zone")
    axes[0].set_ylabel("Average Actual MW")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=zone_error_plot, x="zone_name", y="mae", ax=axes[1])
    axes[1].set_title("Average Absolute Forecast Error by Zone")
    axes[1].set_xlabel("Zone")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=45)

    sns.barplot(data=zone_error_plot, x="zone_name", y="smape_mean", ax=axes[2])
    axes[2].set_title("Average Relative Forecast Error by Zone (sMAPE)")
    axes[2].set_xlabel("Zone")
    axes[2].set_ylabel("sMAPE")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(corr_df: pd.DataFrame, title: str = "Correlation Matrix") -> None:
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.show()
