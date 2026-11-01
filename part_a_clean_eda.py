"""Part A: cleaning, reusable features, data checks, delay evidence, and correlations."""

import os
import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments


def r4(value):
    return round(float(value), 4)


def main():
    os.makedirs("outputs", exist_ok=True)
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    dedup = raw.drop_duplicates().reset_index(drop=True)
    raw_rows = len(raw)
    duplicate_rows_removed = raw_rows - len(dedup)
    missing_distance = int(dedup["distance_km"].isna().sum())
    # Missing-before explicitly excludes sentinel rows, so count only true nulls.
    missing_weight = int(dedup["parcel_weight_kg"].isna().sum())
    sentinel_weight = int((dedup["parcel_weight_kg"] > 100).sum())
    missing_declared = int(dedup["declared_value"].isna().sum())
    missing_payment = int(dedup["payment_mode"].isna().sum())
    spellings_before = int(dedup["service_type"].nunique(dropna=True))

    clean, medians = clean_shipments(raw)
    spellings_after = int(clean["service_type"].nunique(dropna=True))
    model = build_features(clean, hubs)
    model["is_delayed"] = (model["delivery_hours"] > model["promised_hours"]).astype(int)
    model.to_csv("data/modelling_table.csv", index=False)

    q1 = model["delivery_hours"].quantile(0.25)
    q3 = model["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((model["delivery_hours"] < lower) | (model["delivery_hours"] > upper)).sum())

    checks = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", len(clean)),
        ("missing_distance_km_before", missing_distance),
        ("missing_parcel_weight_before", missing_weight),
        ("sentinel_weight_rows", sentinel_weight),
        ("missing_declared_value_before", missing_declared),
        ("missing_payment_mode_before", missing_payment),
        ("service_type_spellings_before", spellings_before),
        ("service_type_spellings_after", spellings_after),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", len(model)),
        ("nulls_in_features_and_target", int(model[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
        ("delayed_count", int(model["is_delayed"].sum())),
        ("delay_rate", model["is_delayed"].mean()),
        ("hours_q1", q1),
        ("hours_q3", q3),
        ("hours_lower_bound", lower),
        ("hours_upper_bound", upper),
        ("hours_outlier_count", outlier_count),
        ("declared_value_mean_inr", model["declared_value_inr"].mean()),
        ("declared_value_median_inr", model["declared_value_inr"].median()),
        ("log_declared_value_mean", model["log_declared_value"].mean()),
        ("log_declared_value_median", model["log_declared_value"].median()),
    ]
    check_df = pd.DataFrame(checks, columns=["metric", "value"])
    check_df["value"] = check_df["value"].map(lambda x: r4(x) if isinstance(x, (float, np.floating)) else x)
    check_df.to_csv("outputs/part_a_checks.csv", index=False)

    model["distance_band"] = pd.cut(
        model["distance_km"], bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"], right=True, include_lowest=True,
    )
    rows = []
    for dimension in ["service_type", "hub_tier", "distance_band"]:
        grouped = model.groupby(dimension, observed=True, sort=True)
        for group, frame in grouped:
            rows.append({
                "dimension": dimension,
                "group": str(group),
                "shipments": len(frame),
                "delayed": int(frame["is_delayed"].sum()),
                "delay_rate": r4(frame["is_delayed"].mean()),
                "mean_hours": r4(frame["delivery_hours"].mean()),
            })
    pd.DataFrame(rows).to_csv("outputs/part_a_delay_by_group.csv", index=False)

    numeric = model.drop(columns=["distance_band"]).select_dtypes(include=[np.number])
    corr = numeric.corr().round(4)
    corr.index.name = "feature"
    corr.reset_index().to_csv("outputs/part_a_correlations.csv", index=False)


if __name__ == "__main__":
    main()
