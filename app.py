from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_processing import aggregate_geolocations, build_state_geo_reference, load_raw_datasets
from feature_engineering import DAY_ORDER, MONTH_ORDER


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DATA_PATH = ARTIFACTS_DIR / "processed_delay_dataset.csv"
RESULTS_PATH = ARTIFACTS_DIR / "model_comparison.csv"
IMPORTANCE_PATH = ARTIFACTS_DIR / "xgboost_feature_importance.csv"
MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "best_model_bundle.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
APP_SUPPORT_PATH = ARTIFACTS_DIR / "app_support_bundle.joblib"
DISTANCE_BAND_ORDER = ["0-50", "50-150", "150-300", "300-600", "600-1000", "1000+"]
AUTO_SELECT_LABEL = "Auto infer from selection"


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    dataset = pd.read_csv(PROCESSED_DATA_PATH)
    results = pd.read_csv(RESULTS_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    return dataset, normalize_results_frame(results), importance, metadata


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    return joblib.load(MODEL_BUNDLE_PATH)


@st.cache_resource(show_spinner=False)
def load_app_support_bundle() -> dict:
    if APP_SUPPORT_PATH.exists():
        return joblib.load(APP_SUPPORT_PATH)
    return {}


@st.cache_data(show_spinner=False)
def load_cleaning_summary(processed_row_count: int, delayed_rate_pct: float) -> pd.DataFrame:
    support_bundle = load_app_support_bundle()
    if support_bundle.get("cleaning_summary_rows"):
        return pd.DataFrame(support_bundle["cleaning_summary_rows"])

    raw = load_raw_datasets()
    orders = raw["orders"]
    customers = raw["customers"]
    sellers = raw["sellers"]
    geolocation = raw["geolocation"]

    delivered_orders = orders.loc[orders["order_status"] == "delivered"].copy()
    usable_target_orders = delivered_orders.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    )

    geo_zip_count = geolocation["geolocation_zip_code_prefix"].nunique()
    customer_geo_coverage = customers["customer_zip_code_prefix"].isin(
        geolocation["geolocation_zip_code_prefix"]
    ).mean() * 100
    seller_geo_coverage = sellers["seller_zip_code_prefix"].isin(
        geolocation["geolocation_zip_code_prefix"]
    ).mean() * 100

    summary_rows = [
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
            "Current result": f"Final model-ready rows: {processed_row_count:,}",
        },
        {
            "Step": "Class definition audit",
            "System action": "Defined `is_delayed = 1` when actual delivery happened after the estimated delivery date.",
            "Current result": f"Delayed shipments in modeling table: {delayed_rate_pct:.2f}%",
        },
    ]
    return pd.DataFrame(summary_rows)


@st.cache_data(show_spinner=False)
def load_location_references() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    support_bundle = load_app_support_bundle()
    if support_bundle:
        geo_by_zip = support_bundle.get("geo_by_zip")
        city_state_reference = support_bundle.get("city_state_reference")
        state_reference = support_bundle.get("state_reference")
        if geo_by_zip is not None and city_state_reference is not None and state_reference is not None:
            return geo_by_zip, city_state_reference, state_reference

    raw = load_raw_datasets()
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
    return geo_by_zip, city_state_reference, state_reference


@st.cache_data(show_spinner=False)
def load_prediction_form_options() -> dict:
    support_bundle = load_app_support_bundle()
    if support_bundle.get("form_options"):
        return support_bundle["form_options"]

    raw = load_raw_datasets()
    customers = raw["customers"].copy()
    sellers = raw["sellers"].copy()

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

    return {
        "customer_zip_by_state": customer_zip_by_state,
        "seller_city_by_state": seller_city_by_state,
        "seller_zip_by_state": seller_zip_by_state,
        "seller_zip_by_state_city": seller_zip_by_state_city,
    }


def artifacts_ready() -> bool:
    required = [PROCESSED_DATA_PATH, RESULTS_PATH, IMPORTANCE_PATH, MODEL_BUNDLE_PATH]
    return all(path.exists() for path in required)


def normalize_results_frame(results: pd.DataFrame) -> pd.DataFrame:
    normalized = results.copy()
    if "threshold" not in normalized.columns and "decision_threshold" in normalized.columns:
        normalized = normalized.rename(columns={"decision_threshold": "threshold"})
    if "threshold" not in normalized.columns:
        normalized["threshold"] = 0.50
    return normalized


def get_prediction_threshold(model_bundle: dict, results: pd.DataFrame) -> float:
    if "best_threshold" in model_bundle:
        return float(model_bundle["best_threshold"])

    best_model_name = model_bundle.get("best_model_name")
    if best_model_name and "threshold" in results.columns:
        for _, row in results.iterrows():
            candidate_name = f"{row['model_name']} | {row['feature_set']}"
            if candidate_name == best_model_name:
                return float(row["threshold"])

    return 0.50


def format_percentage(value: float) -> str:
    return f"{value:.2f}%"


def parse_zip_prefix(raw_value: str) -> int | None:
    digits_only = "".join(character for character in str(raw_value).strip() if character.isdigit())
    if not digits_only:
        return None
    return int(digits_only)


def normalize_city_name(city_name: str) -> str:
    return str(city_name).strip().lower()


def assign_distance_band(distance_km: float) -> str:
    if distance_km <= 50:
        return "0-50"
    if distance_km <= 150:
        return "50-150"
    if distance_km <= 300:
        return "150-300"
    if distance_km <= 600:
        return "300-600"
    if distance_km <= 1000:
        return "600-1000"
    return "1000+"


def _series_mode(series: pd.Series, default_value: object) -> object:
    cleaned = series.dropna()
    if cleaned.empty:
        return default_value
    mode = cleaned.mode()
    if mode.empty:
        return cleaned.iloc[0]
    return mode.iloc[0]


def haversine_distance_km(
    lat_1: float,
    lng_1: float,
    lat_2: float,
    lng_2: float,
) -> float:
    lat_1_rad = np.radians(lat_1)
    lng_1_rad = np.radians(lng_1)
    lat_2_rad = np.radians(lat_2)
    lng_2_rad = np.radians(lng_2)

    delta_lat = lat_2_rad - lat_1_rad
    delta_lng = lng_2_rad - lng_1_rad
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(delta_lng / 2.0) ** 2
    )
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(haversine)))


def resolve_customer_location(
    dataset: pd.DataFrame,
    customer_state: str,
    customer_zip_prefix_text: str,
) -> dict:
    geo_by_zip, _, state_reference = load_location_references()
    zip_prefix = parse_zip_prefix(customer_zip_prefix_text)

    if zip_prefix is not None:
        zip_match = geo_by_zip.loc[geo_by_zip["zip_code_prefix"] == zip_prefix]
        if not zip_match.empty:
            row = zip_match.iloc[0]
            return {
                "customer_lat": float(row["geolocation_lat"]),
                "customer_lng": float(row["geolocation_lng"]),
                "source": "customer ZIP prefix match",
            }

    state_slice = dataset.loc[dataset["customer_state"] == customer_state].copy()
    if not state_slice.empty:
        return {
            "customer_lat": float(pd.to_numeric(state_slice["customer_lat"], errors="coerce").median()),
            "customer_lng": float(pd.to_numeric(state_slice["customer_lng"], errors="coerce").median()),
            "source": "customer state historical median",
        }

    state_match = state_reference.loc[state_reference["state"] == customer_state]
    if not state_match.empty:
        row = state_match.iloc[0]
        return {
            "customer_lat": float(row["state_lat"]),
            "customer_lng": float(row["state_lng"]),
            "source": "customer state centroid fallback",
        }

    return {
        "customer_lat": float(pd.to_numeric(dataset["customer_lat"], errors="coerce").median()),
        "customer_lng": float(pd.to_numeric(dataset["customer_lng"], errors="coerce").median()),
        "source": "global customer coordinate fallback",
    }


def resolve_seller_location(
    dataset: pd.DataFrame,
    seller_state_input: str,
    seller_city_input: str,
    seller_zip_prefix_text: str,
) -> dict:
    geo_by_zip, city_state_reference, state_reference = load_location_references()
    zip_prefix = parse_zip_prefix(seller_zip_prefix_text)
    normalized_city = normalize_city_name(seller_city_input)

    if zip_prefix is not None:
        zip_match = geo_by_zip.loc[geo_by_zip["zip_code_prefix"] == zip_prefix]
        if not zip_match.empty:
            row = zip_match.iloc[0]
            return {
                "seller_state": str(row["geolocation_state"]),
                "seller_city": str(row["geolocation_city"]),
                "seller_lat": float(row["geolocation_lat"]),
                "seller_lng": float(row["geolocation_lng"]),
                "source": "seller ZIP prefix match",
            }

    if normalized_city and seller_state_input:
        city_match = city_state_reference.loc[
            (city_state_reference["state"] == seller_state_input)
            & (city_state_reference["city"] == normalized_city)
        ]
        if not city_match.empty:
            row = city_match.iloc[0]
            return {
                "seller_state": seller_state_input,
                "seller_city": seller_city_input.strip(),
                "seller_lat": float(row["city_lat"]),
                "seller_lng": float(row["city_lng"]),
                "source": "seller city/state centroid match",
            }

    if seller_state_input:
        state_match = state_reference.loc[state_reference["state"] == seller_state_input]
        if not state_match.empty:
            row = state_match.iloc[0]
            return {
                "seller_state": seller_state_input,
                "seller_city": seller_city_input.strip() or "Unknown city",
                "seller_lat": float(row["state_lat"]),
                "seller_lng": float(row["state_lng"]),
                "source": "seller state centroid fallback",
            }

    fallback_state = str(_series_mode(dataset["proxy_seller_state"], "UNK"))
    return {
        "seller_state": fallback_state,
        "seller_city": seller_city_input.strip() or "Unknown city",
        "seller_lat": float(pd.to_numeric(dataset["customer_lat"], errors="coerce").median()),
        "seller_lng": float(pd.to_numeric(dataset["customer_lng"], errors="coerce").median()),
        "source": "global seller coordinate fallback",
    }


def build_prediction_feature_frame(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    base_inputs: dict,
) -> tuple[pd.DataFrame, dict]:
    customer_state = base_inputs["customer_state"]
    customer_location = resolve_customer_location(
        dataset=dataset,
        customer_state=customer_state,
        customer_zip_prefix_text=base_inputs.get("customer_zip_prefix", ""),
    )
    seller_location = resolve_seller_location(
        dataset=dataset,
        seller_state_input=base_inputs.get("seller_state", ""),
        seller_city_input=base_inputs.get("seller_city", ""),
        seller_zip_prefix_text=base_inputs.get("seller_zip_prefix", ""),
    )

    distance_km = haversine_distance_km(
        lat_1=seller_location["seller_lat"],
        lng_1=seller_location["seller_lng"],
        lat_2=customer_location["customer_lat"],
        lng_2=customer_location["customer_lng"],
    )
    distance_band = assign_distance_band(distance_km)

    derived_fields = {
        "proxy_seller_state": seller_location["seller_state"],
        "distance_km": distance_km,
        "distance_band": distance_band,
        "route_pair": f"{customer_state}->{seller_location['seller_state']}",
        "seller_state_match": int(customer_state == seller_location["seller_state"]),
        "customer_lat": customer_location["customer_lat"],
        "customer_lng": customer_location["customer_lng"],
        "seller_city": seller_location["seller_city"],
        "seller_resolution_source": seller_location["source"],
        "customer_resolution_source": customer_location["source"],
    }

    row = {**base_inputs, **derived_fields}
    input_frame = pd.DataFrame([row])

    for feature in feature_columns:
        if feature in input_frame.columns:
            continue

        if feature in dataset.columns:
            feature_series = dataset[feature]
            numeric_series = pd.to_numeric(feature_series, errors="coerce")
            if numeric_series.notna().any():
                input_frame[feature] = float(numeric_series.median())
            else:
                input_frame[feature] = _series_mode(feature_series.astype(str), "UNK")
        else:
            input_frame[feature] = 0

    return input_frame[feature_columns], derived_fields


def render_chart_explanation(what: str, how: str, why: str) -> None:
    st.markdown(
        f"**What it shows:** {what}\n\n"
        f"**How it works:** {how}\n\n"
        f"**Why it matters:** {why}"
    )


def build_filtered_view(dataset: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    month_choices = [month for month in MONTH_ORDER if month in dataset["month"].unique().tolist()]
    day_choices = [day for day in DAY_ORDER if day in dataset["day_of_week"].unique().tolist()]

    selected_months = st.sidebar.multiselect("Month selector", month_choices, default=month_choices)
    selected_days = st.sidebar.multiselect("Day selector", day_choices, default=day_choices)

    max_distance = float(dataset["distance_km"].max())
    selected_distance = st.sidebar.slider(
        "Distance slider (km)",
        min_value=0.0,
        max_value=float(round(max_distance, 1)),
        value=(0.0, float(round(max_distance, 1))),
    )

    filtered = dataset.loc[
        dataset["month"].isin(selected_months)
        & dataset["day_of_week"].isin(selected_days)
        & dataset["distance_km"].between(selected_distance[0], selected_distance[1])
    ].copy()
    return filtered


def render_kpis(filtered: pd.DataFrame) -> None:
    delayed_pct = filtered["is_delayed"].mean() * 100
    avg_delay = filtered["delay_days"].mean()
    median_delay = filtered["delay_days"].median()
    orders_count = len(filtered)

    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("% delayed shipments", format_percentage(delayed_pct))
    col_2.metric("Average delay (days)", f"{avg_delay:.2f}")
    col_3.metric("Median delay (days)", f"{median_delay:.2f}")
    col_4.metric("Orders in view", f"{orders_count:,}")


def render_project_overview(
    metadata: dict,
    model_bundle: dict,
    results: pd.DataFrame,
    cleaning_summary: pd.DataFrame,
) -> None:
    best_model_name = model_bundle.get("best_model_name", "Saved best model")
    threshold = get_prediction_threshold(model_bundle, results)
    target_info = metadata.get("target_definition", {})
    delay_days_text = target_info.get(
        "delay_days",
        "delivery date minus purchase date",
    )
    delayed_text = target_info.get(
        "is_delayed",
        "delivery happens after the estimated delivery date",
    )

    with st.expander("Project Explanation", expanded=True):
        st.write(
            "This project is an end-to-end supply chain delay prediction system built on the Olist Brazilian "
            "e-commerce dataset. It combines order events, customer geography, seller geography, and calendar "
            "signals to estimate whether a shipment will arrive late."
        )
        st.write(
            "The dashboard is designed like an operations analytics layer: it helps a logistics or fulfillment "
            "team monitor delay patterns, compare model strategies, identify risky shipping lanes, and test new "
            "shipment scenarios with the saved best model."
        )
        st.write(
            f"In the current system, `delay_days` is **{delay_days_text}**, and `is_delayed` becomes 1 when "
            f"**{delayed_text}**."
        )
        st.write(
            f"The active best model is **{best_model_name}**, and it applies a probability threshold of "
            f"**{threshold:.2f}** when turning model probabilities into delayed vs on-time decisions."
        )

    with st.expander("Methodology", expanded=True):
        st.write(
            "1. Load and clean the four approved datasets only.\n"
            "2. Filter to delivered orders so the actual final outcome is known.\n"
            "3. Aggregate geolocation records to ZIP-prefix level to reduce duplicate coordinate noise.\n"
            "4. Join orders with customers, then resolve seller geography separately from seller ZIP data.\n"
            "5. Build the target, temporal features, holiday flags, and spatial proxy features.\n"
            "6. Train Logistic Regression and XGBoost systems.\n"
            "7. Compare baseline vs improved feature sets and tune decision thresholds on validation data.\n"
            "8. Deploy the saved best model into the interactive dashboard for monitoring and scenario scoring."
        )
        st.write(
            "The analysis layer is meant to support practical use cases such as SLA monitoring, geographic risk "
            "prioritization, carrier escalation, shipment planning, and seasonal trend review."
        )

    with st.expander("Current System Limitations", expanded=True):
        st.write(
            "The most important limitation is structural: the approved dataset subset does not include an "
            "order-level `seller_id` bridge, so the distance feature is a geographic proxy rather than a true "
            "origin-to-destination shipment distance."
        )
        st.write(
            "This system predicts delay risk, not an exact delivery date. It is more suitable for operational "
            "triage and risk ranking than for a production ETA engine."
        )
        st.write(
            "Additional operational variables such as carrier capacity, warehouse inventory, payment approval "
            "friction, product category, and order-item level seller mapping are intentionally excluded because "
            "they are outside the approved file set."
        )
        st.write(
            "Geolocation is also approximate because ZIP-prefix aggregation smooths many raw coordinates into a "
            "single reference point."
        )

    with st.expander("Dataset Cleaning Summary", expanded=True):
        st.dataframe(cleaning_summary, use_container_width=True, hide_index=True)


def render_lane_risk_heatmap(filtered: pd.DataFrame) -> None:
    st.subheader("Industry Operations Heatmap")
    st.caption(
        "This lane-risk view mimics a logistics control tower. It surfaces high-risk shipping corridors "
        "by combining seller-region proxy, customer region, and distance band."
    )

    if {"proxy_seller_state", "customer_state", "distance_band", "late_days_vs_estimate"}.difference(
        filtered.columns
    ):
        st.info("Lane heatmap is unavailable because the required lane features are missing.")
        return

    metric_option = st.selectbox(
        "Operational heatmap metric",
        [
            "Delayed shipment rate (%)",
            "Average lateness vs ETA (days)",
            "Order volume",
        ],
        index=0,
    )

    working = filtered.copy()
    working["shipping_lane"] = (
        working["proxy_seller_state"].astype(str) + " -> " + working["customer_state"].astype(str)
    )

    distance_band_order = ["0-50", "50-150", "150-300", "300-600", "600-1000", "1000+"]
    top_lanes = working["shipping_lane"].value_counts().head(12).index.tolist()
    lane_subset = working.loc[working["shipping_lane"].isin(top_lanes)].copy()

    if lane_subset.empty:
        st.info("Not enough filtered data is available to build the lane heatmap.")
        return

    summary = (
        lane_subset.groupby(["shipping_lane", "distance_band"], observed=False)
        .agg(
            delayed_rate=("is_delayed", lambda values: float(values.mean() * 100)),
            avg_late_days=("late_days_vs_estimate", "mean"),
            orders=("order_id", "count"),
        )
        .reset_index()
    )

    metric_map = {
        "Delayed shipment rate (%)": ("delayed_rate", "Delayed shipments (%)", ".1f"),
        "Average lateness vs ETA (days)": ("avg_late_days", "Average lateness vs ETA (days)", ".2f"),
        "Order volume": ("orders", "Orders", ".0f"),
    }
    metric_column, color_label, text_format = metric_map[metric_option]

    z_matrix = (
        summary.pivot(index="shipping_lane", columns="distance_band", values=metric_column)
        .reindex(index=top_lanes, columns=distance_band_order)
    )
    delayed_rate_matrix = (
        summary.pivot(index="shipping_lane", columns="distance_band", values="delayed_rate")
        .reindex(index=top_lanes, columns=distance_band_order)
    )
    late_days_matrix = (
        summary.pivot(index="shipping_lane", columns="distance_band", values="avg_late_days")
        .reindex(index=top_lanes, columns=distance_band_order)
    )
    orders_matrix = (
        summary.pivot(index="shipping_lane", columns="distance_band", values="orders")
        .reindex(index=top_lanes, columns=distance_band_order)
    )

    text_matrix = z_matrix.copy().astype(object)
    for row_label in z_matrix.index:
        for col_label in z_matrix.columns:
            cell_value = z_matrix.loc[row_label, col_label]
            if pd.isna(cell_value):
                text_matrix.loc[row_label, col_label] = ""
            else:
                formatted_value = format(cell_value, text_format)
                order_count = orders_matrix.loc[row_label, col_label]
                text_matrix.loc[row_label, col_label] = f"{formatted_value}<br>n={int(order_count)}"

    heatmap = go.Figure(
        data=go.Heatmap(
            z=z_matrix.to_numpy(),
            x=distance_band_order,
            y=top_lanes,
            text=text_matrix.to_numpy(),
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="YlOrRd",
            colorbar={"title": color_label},
            customdata=np.dstack(
                [
                    delayed_rate_matrix.to_numpy(dtype=float),
                    late_days_matrix.to_numpy(dtype=float),
                    orders_matrix.to_numpy(dtype=float),
                ]
            ),
            hovertemplate=(
                "Lane: %{y}<br>"
                "Distance band: %{x}<br>"
                + color_label
                + ": %{z:.2f}<br>"
                "Delayed shipment rate: %{customdata[0]:.2f}%<br>"
                "Average lateness vs ETA: %{customdata[1]:.2f} days<br>"
                "Orders: %{customdata[2]:.0f}<extra></extra>"
            ),
        )
    )
    heatmap.update_layout(
        title="Lane Risk Heatmap: Seller Region -> Customer Region",
        xaxis_title="Distance band (km)",
        yaxis_title="Shipping lane",
        height=650,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    st.plotly_chart(heatmap, use_container_width=True)

    st.caption(
        "Redder cells represent operational hotspots. In a real supply-chain workflow, those lanes are the ones "
        "to inspect for carrier capacity, SLA changes, allocation shifts, or exception handling."
    )
    render_chart_explanation(
        what=(
            "This heatmap highlights which shipping lanes and distance bands are creating the most operational "
            "risk, whether you define risk as delay rate, lateness severity, or shipment volume."
        ),
        how=(
            "The chart groups filtered orders by `proxy_seller_state -> customer_state` and `distance_band`, then "
            "aggregates the selected metric into each cell. Darker cells indicate more critical operating zones."
        ),
        why=(
            "This matters because supply-chain teams manage lanes, not just individual parcels. A lane-level heatmap "
            "helps identify where to change routing strategy, allocate contingency capacity, or escalate service issues."
        ),
    )


def render_analysis_tab(filtered: pd.DataFrame, results: pd.DataFrame, importance: pd.DataFrame) -> None:
    histogram = px.histogram(
        filtered,
        x="delay_days",
        nbins=50,
        title="Delay distribution",
        labels={"delay_days": "Purchase-to-delivery days"},
        color_discrete_sequence=["#1f77b4"],
    )
    histogram.update_layout(bargap=0.05)

    scatter_sample = filtered.sample(n=min(len(filtered), 4000), random_state=42)
    scatter = px.scatter(
        scatter_sample,
        x="distance_km",
        y="delay_days",
        color=scatter_sample["is_delayed"].map({0: "On time", 1: "Delayed"}),
        title="Distance vs delay",
        labels={
            "distance_km": "Seller-network centroid distance (km)",
            "delay_days": "Purchase-to-delivery days",
        },
        opacity=0.5,
    )

    day_summary = (
        filtered.groupby("day_of_week", observed=False)["is_delayed"]
        .mean()
        .reindex(DAY_ORDER)
        .mul(100)
        .reset_index(name="delayed_rate")
    )
    day_chart = px.bar(
        day_summary,
        x="day_of_week",
        y="delayed_rate",
        title="Delay by day of week",
        labels={"delayed_rate": "Delayed shipments (%)", "day_of_week": "Day"},
        color_discrete_sequence=["#ff7f0e"],
    )

    month_summary = (
        filtered.groupby(["month_num", "month"], observed=False)["is_delayed"]
        .mean()
        .mul(100)
        .reset_index(name="delayed_rate")
        .sort_values("month_num")
    )
    month_chart = px.line(
        month_summary,
        x="month",
        y="delayed_rate",
        markers=True,
        title="Monthly delay trend",
        labels={"delayed_rate": "Delayed shipments (%)", "month": "Month"},
    )

    comparison = results.copy()
    comparison["experiment"] = comparison["model_name"] + " | " + comparison["feature_set"]
    model_chart = px.bar(
        comparison,
        x="experiment",
        y="f1_score",
        color="model_name",
        title="Model comparison",
        labels={"f1_score": "F1 score", "experiment": "Experiment"},
        text_auto=".3f",
    )

    importance_chart = px.bar(
        importance.head(15).sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        title="XGBoost feature importance",
        labels={"importance": "Importance", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Sunset",
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(histogram, use_container_width=True)
        render_chart_explanation(
            what=(
                "The histogram shows the full distribution of purchase-to-delivery duration across the filtered order set."
            ),
            how=(
                "Each bar counts how many orders fall into a delivery-duration bucket, making it easy to see the typical "
                "fulfillment window and the long tail of slow shipments."
            ),
            why=(
                "This matters because operations leaders need to know whether delays are isolated exceptions or whether the "
                "whole network is drifting toward slower service."
            ),
        )
    with right:
        st.plotly_chart(scatter, use_container_width=True)
        render_chart_explanation(
            what=(
                "This scatter plot compares shipment distance proxy with actual delivery duration and colors points by delay status."
            ),
            how=(
                "Each point is an order. The x-axis represents the geographic distance proxy, the y-axis represents delivery "
                "duration, and color separates on-time vs delayed outcomes."
            ),
            why=(
                "This matters because it helps validate whether longer lanes are systematically associated with operational delay risk."
            ),
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(day_chart, use_container_width=True)
        render_chart_explanation(
            what=(
                "This bar chart shows how the delayed-shipment rate changes by the day of the week when an order was purchased."
            ),
            how=(
                "For each weekday, the chart calculates the share of filtered orders with `is_delayed = 1` and expresses it as a percentage."
            ),
            why=(
                "This matters because weekday effects often reveal cut-off timing, weekend backlog buildup, staffing imbalance, or dispatch timing issues."
            ),
        )
    with right:
        st.plotly_chart(month_chart, use_container_width=True)
        render_chart_explanation(
            what=(
                "This line chart tracks how the delayed-shipment rate moves across the calendar months in the filtered view."
            ),
            how=(
                "It aggregates the delay flag at the monthly level and then plots the resulting percentage trend in calendar order."
            ),
            why=(
                "This matters because seasonality is a real logistics driver. Peaks can indicate holiday pressure, weather exposure, or capacity bottlenecks."
            ),
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(model_chart, use_container_width=True)
        render_chart_explanation(
            what=(
                "This comparison chart shows how the baseline and advanced models perform, including the effect of adding distance-based features."
            ),
            how=(
                "Each bar summarizes the F1 score for one experiment, combining the model family and whether the distance feature set was used."
            ),
            why=(
                "This matters because operational ML systems should justify added complexity. The chart shows whether the improved system really outperforms the baseline."
            ),
        )
    with right:
        st.plotly_chart(importance_chart, use_container_width=True)
        render_chart_explanation(
            what=(
                "This feature-importance view ranks the variables that most influence XGBoost predictions in the saved model workflow."
            ),
            how=(
                "XGBoost assigns an importance score to each transformed input feature based on how useful it is during tree splits."
            ),
            why=(
                "This matters because decision-makers need interpretability. It reveals whether the model is learning useful business signals such as geography, timing, and planning horizon."
            ),
        )

    render_lane_risk_heatmap(filtered)

    st.caption(
        "Model metrics below use the saved decision threshold for each experiment. "
        "That threshold converts predicted probabilities into delayed vs on-time classes."
    )
    st.dataframe(
        results[
            [
                "model_name",
                "feature_set",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
            ]
        ],
        use_container_width=True,
    )


def render_prediction_tab(model_bundle: dict, dataset: pd.DataFrame) -> None:
    st.subheader("Delay probability prediction")
    st.caption(
        "The saved best model powers this form. Instead of entering distance manually, you can provide seller "
        "address details and the system will estimate the geographic distance automatically."
    )
    st.write(
        "Some model features are engineered behind the scenes. When you submit the form, the app automatically "
        "derives lane-related proxy fields from historical patterns in the processed dataset so the saved model "
        "receives the full feature set it was trained on."
    )

    model = model_bundle["model"]
    feature_columns = model_bundle["best_feature_columns"]
    results = pd.DataFrame(model_bundle.get("results", []))
    threshold = get_prediction_threshold(model_bundle, normalize_results_frame(results))
    customer_states = sorted(dataset["customer_state"].dropna().unique().tolist())
    seller_states = sorted(dataset["proxy_seller_state"].dropna().astype(str).unique().tolist())
    form_options = load_prediction_form_options()

    with st.form("prediction_form"):
        col_1, col_2, col_3 = st.columns(3)
        promised_lead_days = col_1.number_input(
            "Promised lead days", min_value=0.0, value=12.0, step=0.5
        )
        purchase_hour = col_2.slider("Purchase hour", min_value=0, max_value=23, value=12)
        customer_state = col_3.selectbox("Customer state", customer_states, index=0)

        customer_zip_choices = [AUTO_SELECT_LABEL] + form_options["customer_zip_by_state"].get(
            customer_state,
            [],
        )

        col_4, col_5, col_6 = st.columns(3)
        day_of_week = col_4.selectbox("Day of week", DAY_ORDER, index=0)
        month = col_5.selectbox("Month", MONTH_ORDER, index=0)
        customer_zip_prefix = col_6.selectbox("Customer ZIP prefix", customer_zip_choices, index=0)

        col_7, col_8, col_9 = st.columns(3)
        seller_state = col_9.selectbox("Seller state", seller_states, index=0)
        seller_city_choices = [AUTO_SELECT_LABEL] + form_options["seller_city_by_state"].get(
            seller_state,
            [],
        )
        seller_city = col_8.selectbox("Seller city", seller_city_choices, index=0)

        if seller_city != AUTO_SELECT_LABEL:
            seller_zip_pool = form_options["seller_zip_by_state_city"].get((seller_state, seller_city), [])
        else:
            seller_zip_pool = form_options["seller_zip_by_state"].get(seller_state, [])
        seller_zip_choices = [AUTO_SELECT_LABEL] + seller_zip_pool
        seller_zip_prefix = col_7.selectbox("Seller ZIP prefix", seller_zip_choices, index=0)

        is_holiday = st.checkbox("Holiday purchase", value=False)
        submit = st.form_submit_button("Predict delay")

    if not submit:
        return

    base_inputs = {
        "promised_lead_days": promised_lead_days,
        "purchase_hour": purchase_hour,
        "is_holiday": int(is_holiday),
        "is_weekend": int(day_of_week in {"Saturday", "Sunday"}),
        "day_of_week": day_of_week,
        "month": month,
        "customer_state": customer_state,
        "customer_zip_prefix": "" if customer_zip_prefix == AUTO_SELECT_LABEL else customer_zip_prefix,
        "seller_zip_prefix": "" if seller_zip_prefix == AUTO_SELECT_LABEL else seller_zip_prefix,
        "seller_city": "" if seller_city == AUTO_SELECT_LABEL else seller_city,
        "seller_state": seller_state,
    }
    input_frame, derived_fields = build_prediction_feature_frame(
        dataset=dataset,
        feature_columns=feature_columns,
        base_inputs=base_inputs,
    )

    probability = model.predict_proba(input_frame[feature_columns])[:, 1][0]
    label = "Delayed" if probability >= threshold else "On time"
    confidence = "High" if probability >= 0.75 or probability <= 0.25 else "Medium"

    st.metric("Predicted delay probability", format_percentage(probability * 100))
    st.write(f"Predicted class at threshold {threshold:.2f}: **{label}**")
    st.write(f"Confidence: **{confidence}**")
    st.write(
        f"Estimated geographic distance used by the model: **{derived_fields['distance_km']:.1f} km**"
    )
    st.caption(
        "The app estimated seller and customer coordinates from the selected location fields before scoring the shipment."
    )


def main() -> None:
    st.set_page_config(page_title="Supply Chain Delay Prediction", layout="wide")
    st.title("Supply Chain Delay Prediction Dashboard")
    st.caption(
        "Olist Brazilian e-commerce analysis with comparative modeling, interactive filters, and live prediction."
    )

    if not artifacts_ready():
        st.warning(
            "Artifacts are missing. Run the training pipeline to generate the processed dataset, metrics, and model bundle."
        )
        if st.button("Run full pipeline"):
            with st.spinner("Training models and generating artifacts..."):
                from model_training import run_training_pipeline

                run_training_pipeline()
                st.cache_data.clear()
                st.cache_resource.clear()
            st.success("Artifacts generated. Reload the page to explore the dashboard.")
        st.stop()

    dataset, results, importance, metadata = load_artifacts()
    model_bundle = load_model_bundle()
    cleaning_summary = load_cleaning_summary(
        processed_row_count=len(dataset),
        delayed_rate_pct=float(dataset["is_delayed"].mean() * 100),
    )

    st.info(model_bundle["distance_proxy_note"])
    render_project_overview(metadata, model_bundle, results, cleaning_summary)
    filtered = build_filtered_view(dataset)
    if filtered.empty:
        st.warning("No rows match the current filters. Adjust the selections in the sidebar.")
        st.stop()

    render_kpis(filtered)
    analysis_tab, predict_tab = st.tabs(["Analysis", "Predict"])

    with analysis_tab:
        render_analysis_tab(filtered, results, importance)

    with predict_tab:
        render_prediction_tab(model_bundle, dataset)


if __name__ == "__main__":
    main()
