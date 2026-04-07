from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data_processing import (
    DISTANCE_PROXY_NOTE,
    RAW_DATA_DIR,
    aggregate_geolocations,
    build_model_base_table,
    build_state_geo_reference,
    load_raw_datasets,
)
from evaluation import (
    compute_classification_metrics,
    ensure_output_dir,
    plot_delay_by_day_of_week,
    plot_delay_distribution,
    plot_distance_vs_delay,
    plot_feature_importance,
    plot_lane_risk_heatmap,
    plot_model_comparison,
    plot_monthly_delay_trend,
    save_confusion_matrix_plot,
)
from feature_engineering import create_features, get_feature_sets


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
APP_SUPPORT_BUNDLE_PATH = ARTIFACTS_DIR / "app_support_bundle.joblib"
RANDOM_STATE = 42


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_model_specs(positive_class_weight: float) -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=800,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=1.0,
            scale_pos_weight=positive_class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def derive_feature_types(feature_columns: List[str]) -> Tuple[List[str], List[str]]:
    numeric_features = [
        feature
        for feature in feature_columns
        if feature
        in {
            "promised_lead_days",
            "purchase_hour",
            "is_holiday",
            "is_weekend",
            "distance_km",
            "seller_state_match",
            "customer_lat",
            "customer_lng",
        }
    ]
    categorical_features = [feature for feature in feature_columns if feature not in numeric_features]
    return numeric_features, categorical_features


def extract_xgboost_feature_importance(model_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]
    importance_frame = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_frame["feature"] = (
        importance_frame["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("encoder__", "", regex=False)
    )
    return importance_frame


def select_best_threshold(y_true: pd.Series, y_score: pd.Series) -> float:
    candidate_thresholds = [threshold / 100 for threshold in range(10, 81, 2)]
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        predictions = (pd.Series(y_score) >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold


def build_app_support_bundle(
    data_dir: Path | str = RAW_DATA_DIR,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
    processed_row_count: int | None = None,
    delayed_rate_pct: float | None = None,
) -> dict:
    artifacts_path = ensure_output_dir(artifacts_dir)
    raw = load_raw_datasets(data_dir)
    orders = raw["orders"]
    customers = raw["customers"].copy()
    sellers = raw["sellers"].copy()
    geolocation = raw["geolocation"].copy()

    geolocation["geolocation_city"] = (
        geolocation["geolocation_city"].astype(str).str.strip().str.lower()
    )
    geo_by_zip = aggregate_geolocations(geolocation)
    city_state_reference = (
        geolocation.groupby(["geolocation_state", "geolocation_city"], as_index=False)
        .agg(
            city_lat=("geolocation_lat", "mean"),
            city_lng=("geolocation_lng", "mean"),
        )
        .rename(
            columns={
                "geolocation_state": "state",
                "geolocation_city": "city",
            }
        )
    )
    state_reference = build_state_geo_reference(geolocation).rename(
        columns={"geolocation_state": "state"}
    )

    customers["customer_state"] = customers["customer_state"].astype(str)
    customers["customer_zip_code_prefix"] = (
        customers["customer_zip_code_prefix"].astype("Int64").astype(str)
    )

    sellers["seller_state"] = sellers["seller_state"].astype(str)
    sellers["seller_city"] = sellers["seller_city"].astype(str).str.strip()
    sellers["seller_zip_code_prefix"] = sellers["seller_zip_code_prefix"].astype("Int64").astype(str)

    customer_zip_by_state: dict[str, list[str]] = {}
    for state, group in customers.groupby("customer_state"):
        customer_zip_by_state[state] = sorted(group["customer_zip_code_prefix"].dropna().unique().tolist())

    seller_city_by_state: dict[str, list[str]] = {}
    seller_zip_by_state: dict[str, list[str]] = {}
    seller_zip_by_state_city: dict[tuple[str, str], list[str]] = {}

    for state, group in sellers.groupby("seller_state"):
        seller_city_by_state[state] = sorted(group["seller_city"].dropna().unique().tolist())
        seller_zip_by_state[state] = sorted(group["seller_zip_code_prefix"].dropna().unique().tolist())

    for (state, city), group in sellers.groupby(["seller_state", "seller_city"]):
        seller_zip_by_state_city[(state, city)] = sorted(
            group["seller_zip_code_prefix"].dropna().unique().tolist()
        )

    delivered_orders = orders.loc[orders["order_status"] == "delivered"].copy()
    usable_target_orders = delivered_orders.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    )

    geo_zip_count = geolocation["geolocation_zip_code_prefix"].nunique()
    customer_geo_coverage = raw["customers"]["customer_zip_code_prefix"].isin(
        geolocation["geolocation_zip_code_prefix"]
    ).mean() * 100
    seller_geo_coverage = raw["sellers"]["seller_zip_code_prefix"].isin(
        geolocation["geolocation_zip_code_prefix"]
    ).mean() * 100

    processed_dataset_path = Path(artifacts_path) / "processed_delay_dataset.csv"
    if (processed_row_count is None or delayed_rate_pct is None) and processed_dataset_path.exists():
        processed_target = pd.read_csv(processed_dataset_path, usecols=["is_delayed"])
        if processed_row_count is None:
            processed_row_count = int(len(processed_target))
        if delayed_rate_pct is None:
            delayed_rate_pct = float(processed_target["is_delayed"].mean() * 100)

    final_processed_row_count = processed_row_count if processed_row_count is not None else int(len(usable_target_orders))
    final_delayed_rate_pct = delayed_rate_pct if delayed_rate_pct is not None else 0.0

    cleaning_summary_rows = [
        {
            "Step": "Raw source intake",
            "System action": "Loaded only the four approved Olist files: orders, customers, sellers, and geolocation.",
            "Current result": (
                f"Orders: {len(orders):,} | Customers: {len(customers):,} | "
                f"Sellers: {len(sellers):,} | Geolocation rows: {len(geolocation):,}"
            ),
        },
        {
            "Step": "Order filtering",
            "System action": "Restricted the modeling base to delivered orders so the final delivery outcome is known.",
            "Current result": f"Delivered orders retained: {len(delivered_orders):,}",
        },
        {
            "Step": "Target completeness",
            "System action": (
                "Dropped delivered orders missing purchase timestamp, delivered customer timestamp, "
                "or estimated delivery timestamp."
            ),
            "Current result": f"Target-ready delivered orders: {len(usable_target_orders):,}",
        },
        {
            "Step": "Geolocation standardization",
            "System action": "Collapsed noisy geolocation rows into ZIP-prefix level reference coordinates.",
            "Current result": f"Unique geolocation ZIP prefixes after aggregation: {geo_zip_count:,}",
        },
        {
            "Step": "Customer coordinate coverage",
            "System action": "Mapped customer ZIP prefixes to geolocation records and used state medians as fallback.",
            "Current result": f"Customer ZIP coverage from geolocation table: {customer_geo_coverage:.2f}%",
        },
        {
            "Step": "Seller coordinate coverage",
            "System action": "Mapped seller ZIP prefixes to geolocation records and used state medians as fallback.",
            "Current result": f"Seller ZIP coverage from geolocation table: {seller_geo_coverage:.2f}%",
        },
        {
            "Step": "Spatial proxy feature",
            "System action": (
                "Created `distance_km` using a nearest-seller geographic proxy because order-level seller links "
                "are not available in the approved file subset."
            ),
            "Current result": "Distance feature available for downstream analysis and modeling.",
        },
        {
            "Step": "Feature quality filtering",
            "System action": "Removed rows with invalid negative delivery durations or negative promised lead times.",
            "Current result": f"Final model-ready rows: {final_processed_row_count:,}",
        },
        {
            "Step": "Class definition audit",
            "System action": "Defined `is_delayed = 1` when actual delivery happened after the estimated delivery date.",
            "Current result": f"Delayed shipments in modeling table: {final_delayed_rate_pct:.2f}%",
        },
    ]

    support_bundle = {
        "cleaning_summary_rows": cleaning_summary_rows,
        "geo_by_zip": geo_by_zip,
        "city_state_reference": city_state_reference,
        "state_reference": state_reference,
        "form_options": {
            "customer_zip_by_state": customer_zip_by_state,
            "seller_city_by_state": seller_city_by_state,
            "seller_zip_by_state": seller_zip_by_state,
            "seller_zip_by_state_city": seller_zip_by_state_city,
        },
    }
    joblib.dump(support_bundle, Path(artifacts_path) / APP_SUPPORT_BUNDLE_PATH.name)
    return support_bundle


def run_training_pipeline(
    data_dir: Path | str = RAW_DATA_DIR,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> dict:
    artifacts_path = ensure_output_dir(artifacts_dir)
    figures_path = ensure_output_dir(Path(artifacts_path) / "figures")

    base_table = build_model_base_table(data_dir)
    feature_frame = create_features(base_table)
    feature_frame.to_csv(artifacts_path / "processed_delay_dataset.csv", index=False)

    baseline_features, enhanced_features = get_feature_sets()
    train_valid_frame, test_frame = train_test_split(
        feature_frame,
        test_size=0.2,
        stratify=feature_frame["is_delayed"],
        random_state=RANDOM_STATE,
    )
    train_frame, validation_frame = train_test_split(
        train_valid_frame,
        test_size=0.2,
        stratify=train_valid_frame["is_delayed"],
        random_state=RANDOM_STATE,
    )

    y_train = train_frame["is_delayed"]
    y_test = test_frame["is_delayed"]
    positive_class_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    feature_variants = {
        "Without Distance": baseline_features,
        "With Distance": enhanced_features,
    }
    model_specs = build_model_specs(positive_class_weight)

    results = []
    trained_models: Dict[str, Pipeline] = {}
    selected_thresholds: Dict[str, float] = {}

    for feature_set_name, feature_columns in feature_variants.items():
        numeric_features, categorical_features = derive_feature_types(feature_columns)
        X_train = train_frame[feature_columns]
        X_validation = validation_frame[feature_columns]
        X_train_valid = train_valid_frame[feature_columns]
        X_test = test_frame[feature_columns]

        for model_name, estimator in model_specs.items():
            model_pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        build_preprocessor(
                            numeric_features,
                            categorical_features,
                            scale_numeric=(model_name == "Logistic Regression"),
                        ),
                    ),
                    ("classifier", clone(estimator)),
                ]
            )
            model_pipeline.fit(X_train, y_train)
            validation_probabilities = model_pipeline.predict_proba(X_validation)[:, 1]
            decision_threshold = select_best_threshold(
                validation_frame["is_delayed"], validation_probabilities
            )

            model_pipeline.fit(X_train_valid, train_valid_frame["is_delayed"])
            probabilities = model_pipeline.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= decision_threshold).astype(int)

            metrics = compute_classification_metrics(y_test, predictions, probabilities)
            results.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "feature_columns": feature_columns,
                    "decision_threshold": decision_threshold,
                    **metrics,
                }
            )
            model_key = f"{model_name} | {feature_set_name}"
            trained_models[model_key] = model_pipeline
            selected_thresholds[model_key] = decision_threshold

    results_frame = pd.DataFrame(results).sort_values(
        ["f1_score", "recall", "precision"],
        ascending=False,
    )
    results_frame.to_csv(artifacts_path / "model_comparison.csv", index=False)
    results_frame.to_json(artifacts_path / "model_comparison.json", orient="records", indent=2)

    best_row = results_frame.iloc[0].to_dict()
    best_model_key = f"{best_row['model_name']} | {best_row['feature_set']}"
    best_model = trained_models[best_model_key]
    best_feature_columns = list(best_row["feature_columns"])
    best_threshold = float(selected_thresholds[best_model_key])
    best_probabilities = best_model.predict_proba(test_frame[best_feature_columns])[:, 1]
    best_predictions = (best_probabilities >= best_threshold).astype(int)

    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=best_predictions,
        output_path=figures_path / "best_model_confusion_matrix.png",
        title=f"Confusion Matrix: {best_model_key}",
    )

    xgb_with_distance = trained_models["XGBoost | With Distance"]
    importance_frame = extract_xgboost_feature_importance(xgb_with_distance)
    importance_frame.to_csv(artifacts_path / "xgboost_feature_importance.csv", index=False)

    plot_delay_distribution(feature_frame, figures_path / "delay_distribution.png")
    plot_distance_vs_delay(feature_frame, figures_path / "distance_vs_delay.png")
    plot_delay_by_day_of_week(feature_frame, figures_path / "delay_by_day_of_week.png")
    plot_monthly_delay_trend(feature_frame, figures_path / "monthly_delay_trend.png")
    plot_model_comparison(results_frame, figures_path / "model_comparison.png")
    plot_feature_importance(importance_frame, figures_path / "xgboost_feature_importance.png")
    plot_lane_risk_heatmap(feature_frame, figures_path / "lane_risk_heatmap.png")

    metadata = {
        "distance_proxy_note": DISTANCE_PROXY_NOTE,
        "rows_modeled": int(len(feature_frame)),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "best_model": best_model_key,
        "best_model_f1": float(best_row["f1_score"]),
        "best_threshold": best_threshold,
        "target_definition": {
            "delay_days": "order_delivered_customer_date - order_purchase_timestamp",
            "is_delayed": "order_delivered_customer_date > order_estimated_delivery_date",
        },
    }
    with open(artifacts_path / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    model_bundle = {
        "model": best_model,
        "best_model_name": best_model_key,
        "best_feature_columns": best_feature_columns,
        "best_threshold": best_threshold,
        "baseline_features": baseline_features,
        "enhanced_features": enhanced_features,
        "distance_proxy_note": DISTANCE_PROXY_NOTE,
        "feature_importance": importance_frame.to_dict(orient="records"),
        "results": results_frame.to_dict(orient="records"),
    }
    joblib.dump(model_bundle, artifacts_path / "best_model_bundle.joblib")
    build_app_support_bundle(
        data_dir=data_dir,
        artifacts_dir=artifacts_path,
        processed_row_count=int(len(feature_frame)),
        delayed_rate_pct=float(feature_frame["is_delayed"].mean() * 100),
    )

    return {
        "feature_frame": feature_frame,
        "results_frame": results_frame,
        "importance_frame": importance_frame,
        "metadata": metadata,
    }


if __name__ == "__main__":
    output = run_training_pipeline()
    print("Training complete.")
    print(
        output["results_frame"][
            [
                "model_name",
                "feature_set",
                "decision_threshold",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
            ]
        ].to_string(index=False)
    )
    print()
    print(json.dumps(output["metadata"], indent=2))
