from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from feature_engineering import DAY_ORDER


sns.set_theme(style="whitegrid", context="talk")
DISTANCE_BAND_ORDER = ["0-50", "50-150", "150-300", "300-600", "600-1000", "1000+"]


def ensure_output_dir(path: Path | str) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series | None = None,
) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }
    if y_score is not None:
        metrics["avg_predicted_delay_probability"] = float(pd.Series(y_score).mean())
    return metrics


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path | str,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["On time", "Delayed"])
    ax.set_yticklabels(["On time", "Delayed"], rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delay_distribution(feature_frame: pd.DataFrame, output_path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(feature_frame["delay_days"], bins=50, color="#1f77b4", ax=ax)
    ax.set_title("Shipment Delay Distribution")
    ax.set_xlabel("Purchase-to-delivery days")
    ax.set_ylabel("Orders")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distance_vs_delay(feature_frame: pd.DataFrame, output_path: Path | str) -> None:
    sample = feature_frame.sample(n=min(len(feature_frame), 5000), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="distance_km",
        y="delay_days",
        hue="is_delayed",
        palette={0: "#2ca02c", 1: "#d62728"},
        alpha=0.55,
        s=45,
        ax=ax,
    )
    ax.set_title("Distance vs Delay")
    ax.set_xlabel("Seller-network centroid distance (km)")
    ax.set_ylabel("Purchase-to-delivery days")
    ax.legend(title="Delayed", labels=["On time", "Delayed"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delay_by_day_of_week(feature_frame: pd.DataFrame, output_path: Path | str) -> None:
    summary = (
        feature_frame.groupby("day_of_week", observed=False)["is_delayed"]
        .mean()
        .reindex(DAY_ORDER)
        .mul(100)
        .reset_index(name="delayed_rate")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=summary, x="day_of_week", y="delayed_rate", color="#ff7f0e", ax=ax)
    ax.set_title("Delay Rate by Purchase Day")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Delayed shipments (%)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_delay_trend(feature_frame: pd.DataFrame, output_path: Path | str) -> None:
    summary = (
        feature_frame.groupby(["month_num", "month"], observed=False)["is_delayed"]
        .mean()
        .mul(100)
        .reset_index(name="delayed_rate")
        .sort_values("month_num")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=summary, x="month", y="delayed_rate", marker="o", color="#9467bd", ax=ax)
    ax.set_title("Monthly Delay Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Delayed shipments (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results_frame: pd.DataFrame, output_path: Path | str) -> None:
    comparison = results_frame.copy()
    comparison["experiment"] = comparison["model_name"] + " | " + comparison["feature_set"]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=comparison,
        x="experiment",
        y="f1_score",
        hue="experiment",
        palette="viridis",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Model Comparison by F1 Score")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("F1 score")
    ax.tick_params(axis="x", rotation=25)
    for patch, score in zip(ax.patches, comparison["f1_score"]):
        ax.annotate(
            f"{score:.3f}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance_frame: pd.DataFrame, output_path: Path | str) -> None:
    top_features = importance_frame.sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=top_features,
        x="importance",
        y="feature",
        hue="feature",
        palette="magma",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Top XGBoost Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lane_risk_heatmap(feature_frame: pd.DataFrame, output_path: Path | str) -> None:
    working = feature_frame.copy()
    working["shipping_lane"] = (
        working["proxy_seller_state"].astype(str) + " -> " + working["customer_state"].astype(str)
    )
    working["distance_band"] = pd.Categorical(
        working["distance_band"],
        categories=DISTANCE_BAND_ORDER,
        ordered=True,
    )

    top_lanes = working["shipping_lane"].value_counts().head(12).index.tolist()
    lane_subset = working.loc[working["shipping_lane"].isin(top_lanes)].copy()

    if lane_subset.empty:
        return

    heatmap_data = (
        lane_subset.groupby(["shipping_lane", "distance_band"], observed=False)["is_delayed"]
        .mean()
        .mul(100)
        .reset_index(name="delayed_rate")
        .pivot(index="shipping_lane", columns="distance_band", values="delayed_rate")
        .reindex(index=top_lanes, columns=DISTANCE_BAND_ORDER)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Delayed shipments (%)"},
        ax=ax,
    )
    ax.set_title("Operational Lane Risk Heatmap")
    ax.set_xlabel("Distance band (km)")
    ax.set_ylabel("Shipping lane (seller state -> customer state)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
