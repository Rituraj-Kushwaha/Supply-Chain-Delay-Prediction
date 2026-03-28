from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


EARTH_RADIUS_KM = 6371.0088
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_dataset"

# The provided four-file subset does not include an order-level seller mapping.
# We therefore estimate seller-to-customer distance as the distance from each
# customer location to the nearest seller in the marketplace network.
DISTANCE_PROXY_NOTE = (
    "Distance is computed as a nearest-seller proxy because the provided "
    "four-file subset does not contain an order-level seller_id join."
)


def load_raw_datasets(data_dir: Path | str = RAW_DATA_DIR) -> Dict[str, pd.DataFrame]:
    data_path = Path(data_dir)
    orders = pd.read_csv(
        data_path / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    customers = pd.read_csv(data_path / "olist_customers_dataset.csv")
    sellers = pd.read_csv(data_path / "olist_sellers_dataset.csv")
    geolocation = pd.read_csv(data_path / "olist_geolocation_dataset.csv")
    return {
        "orders": orders,
        "customers": customers,
        "sellers": sellers,
        "geolocation": geolocation,
    }


def _safe_mode(series: pd.Series) -> object:
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else np.nan


def aggregate_geolocations(geolocation: pd.DataFrame) -> pd.DataFrame:
    return (
        geolocation.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            geolocation_lat=("geolocation_lat", "mean"),
            geolocation_lng=("geolocation_lng", "mean"),
            geolocation_city=("geolocation_city", _safe_mode),
            geolocation_state=("geolocation_state", _safe_mode),
        )
        .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    )


def build_state_geo_reference(geolocation: pd.DataFrame) -> pd.DataFrame:
    return geolocation.groupby("geolocation_state", as_index=False).agg(
        state_lat=("geolocation_lat", "median"),
        state_lng=("geolocation_lng", "median"),
    )


def prepare_customer_locations(
    customers: pd.DataFrame,
    geo_by_zip: pd.DataFrame,
    state_geo_reference: pd.DataFrame,
) -> pd.DataFrame:
    customer_locations = customers.merge(
        geo_by_zip,
        left_on="customer_zip_code_prefix",
        right_on="zip_code_prefix",
        how="left",
    ).drop(columns=["zip_code_prefix"])

    customer_locations = customer_locations.merge(
        state_geo_reference,
        left_on="customer_state",
        right_on="geolocation_state",
        how="left",
    )

    customer_locations["customer_lat"] = customer_locations["geolocation_lat"].fillna(
        customer_locations["state_lat"]
    )
    customer_locations["customer_lng"] = customer_locations["geolocation_lng"].fillna(
        customer_locations["state_lng"]
    )

    return customer_locations[
        [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
            "customer_lat",
            "customer_lng",
        ]
    ]


def prepare_seller_locations(
    sellers: pd.DataFrame,
    geo_by_zip: pd.DataFrame,
    state_geo_reference: pd.DataFrame,
) -> pd.DataFrame:
    seller_locations = sellers.merge(
        geo_by_zip,
        left_on="seller_zip_code_prefix",
        right_on="zip_code_prefix",
        how="left",
    ).drop(columns=["zip_code_prefix"])

    seller_locations = seller_locations.merge(
        state_geo_reference,
        left_on="seller_state",
        right_on="geolocation_state",
        how="left",
    )

    seller_locations["seller_lat"] = seller_locations["geolocation_lat"].fillna(
        seller_locations["state_lat"]
    )
    seller_locations["seller_lng"] = seller_locations["geolocation_lng"].fillna(
        seller_locations["state_lng"]
    )

    return seller_locations[
        [
            "seller_id",
            "seller_zip_code_prefix",
            "seller_city",
            "seller_state",
            "seller_lat",
            "seller_lng",
        ]
    ]


def attach_seller_distance_proxy(
    orders_with_customers: pd.DataFrame,
    sellers_with_locations: pd.DataFrame,
) -> pd.DataFrame:
    enriched = orders_with_customers.copy()

    valid_sellers = sellers_with_locations.dropna(subset=["seller_lat", "seller_lng"]).reset_index(
        drop=True
    )
    valid_customers = enriched["customer_lat"].notna() & enriched["customer_lng"].notna()

    if valid_sellers.empty or not valid_customers.any():
        enriched["distance_km"] = np.nan
        enriched["proxy_seller_id"] = pd.NA
        enriched["proxy_seller_state"] = "UNK"
        return enriched

    seller_coordinates = np.radians(valid_sellers[["seller_lat", "seller_lng"]].to_numpy(dtype=float))
    customer_coordinates = np.radians(
        enriched.loc[valid_customers, ["customer_lat", "customer_lng"]].to_numpy(dtype=float)
    )
    tree = BallTree(seller_coordinates, metric="haversine")
    distances_rad, indices = tree.query(customer_coordinates, k=1)
    nearest_sellers = valid_sellers.iloc[indices.ravel()].reset_index(drop=True)

    enriched.loc[valid_customers, "distance_km"] = distances_rad.ravel() * EARTH_RADIUS_KM
    enriched.loc[valid_customers, "proxy_seller_id"] = nearest_sellers["seller_id"].to_numpy()
    enriched.loc[valid_customers, "proxy_seller_state"] = nearest_sellers["seller_state"].to_numpy()

    median_distance = enriched["distance_km"].median()
    enriched["distance_km"] = enriched["distance_km"].fillna(median_distance)
    enriched["proxy_seller_state"] = enriched["proxy_seller_state"].fillna("UNK")
    return enriched


def build_model_base_table(data_dir: Path | str = RAW_DATA_DIR) -> pd.DataFrame:
    datasets = load_raw_datasets(data_dir)
    orders = datasets["orders"]
    customers = datasets["customers"]
    sellers = datasets["sellers"]
    geolocation = datasets["geolocation"]

    geo_by_zip = aggregate_geolocations(geolocation)
    state_geo_reference = build_state_geo_reference(geolocation)

    customer_locations = prepare_customer_locations(customers, geo_by_zip, state_geo_reference)
    seller_locations = prepare_seller_locations(sellers, geo_by_zip, state_geo_reference)

    delivered_orders = orders.loc[orders["order_status"] == "delivered"].copy()
    delivered_orders = delivered_orders.merge(customer_locations, on="customer_id", how="left")
    delivered_orders = attach_seller_distance_proxy(delivered_orders, seller_locations)
    delivered_orders["distance_proxy_note"] = DISTANCE_PROXY_NOTE
    return delivered_orders


if __name__ == "__main__":
    prepared = build_model_base_table()
    print(prepared.head().to_string(index=False))
    print()
    print(f"Rows prepared: {len(prepared):,}")
    print(DISTANCE_PROXY_NOTE)
