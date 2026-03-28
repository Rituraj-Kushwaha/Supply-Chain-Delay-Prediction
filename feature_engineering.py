from __future__ import annotations

from typing import Tuple

import holidays
import numpy as np
import pandas as pd


DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MONTH_ORDER = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def create_features(base_table: pd.DataFrame) -> pd.DataFrame:
    df = base_table.copy()
    df = df.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "customer_state",
        ]
    ).copy()

    purchase_ts = df["order_purchase_timestamp"]
    delivered_ts = df["order_delivered_customer_date"]
    estimated_ts = df["order_estimated_delivery_date"]

    df["delay_days"] = (delivered_ts - purchase_ts).dt.total_seconds() / 86400.0
    df["late_days_vs_estimate"] = (delivered_ts - estimated_ts).dt.total_seconds() / 86400.0
    df["is_delayed"] = (df["late_days_vs_estimate"] > 0).astype(int)

    df["day_of_week"] = purchase_ts.dt.day_name()
    df["month"] = purchase_ts.dt.strftime("%b")
    df["month_num"] = purchase_ts.dt.month
    df["purchase_hour"] = purchase_ts.dt.hour
    df["is_weekend"] = purchase_ts.dt.dayofweek.isin([5, 6]).astype(int)
    df["promised_lead_days"] = (estimated_ts - purchase_ts).dt.total_seconds() / 86400.0

    holiday_calendar = holidays.country_holidays(
        "BR", years=sorted(df["order_purchase_timestamp"].dt.year.unique().tolist())
    )
    df["is_holiday"] = purchase_ts.dt.date.map(lambda value: int(value in holiday_calendar))

    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=DAY_ORDER, ordered=True)
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
    df["distance_km"] = df["distance_km"].astype(float)
    df["proxy_seller_state"] = df["proxy_seller_state"].fillna("UNK").astype(str)
    df["route_pair"] = df["customer_state"].astype(str) + "->" + df["proxy_seller_state"]
    df["seller_state_match"] = (df["customer_state"] == df["proxy_seller_state"]).astype(int)

    df["distance_band"] = pd.cut(
        df["distance_km"],
        bins=[0, 50, 150, 300, 600, 1000, np.inf],
        labels=["0-50", "50-150", "150-300", "300-600", "600-1000", "1000+"],
        include_lowest=True,
    )

    df = df[(df["delay_days"] >= 0) & (df["promised_lead_days"] >= 0)].copy()
    return df


def get_feature_sets() -> Tuple[list[str], list[str]]:
    baseline_features = [
        "promised_lead_days",
        "purchase_hour",
        "is_holiday",
        "is_weekend",
        "day_of_week",
        "month",
        "customer_state",
    ]
    enhanced_features = baseline_features + [
        "distance_km",
        "proxy_seller_state",
        "distance_band",
        "route_pair",
        "seller_state_match",
        "customer_lat",
        "customer_lng",
    ]
    return baseline_features, enhanced_features


def split_train_test_chronologically(
    feature_frame: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = feature_frame.sort_values("order_purchase_timestamp").reset_index(drop=True)
    split_index = int(len(ordered) * train_fraction)
    train_frame = ordered.iloc[:split_index].copy()
    test_frame = ordered.iloc[split_index:].copy()
    return train_frame, test_frame
