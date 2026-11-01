"""Rebuild and package the final delay-classification pipeline."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    max_depth_raw = str(config["max_depth"])
    max_depth = None if max_depth_raw == "unlimited" else int(float(max_depth_raw))
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
        ("model", RandomForestClassifier(
            random_state=42,
            n_estimators=int(config["n_estimators"]),
            max_depth=max_depth,
            min_samples_leaf=int(config["min_samples_leaf"]),
        )),
    ])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "models/delay_model.pkl")
    pd.DataFrame({"threshold": [round(float(config["threshold"]), 4)]}).to_csv(
        "models/delay_threshold.csv", index=False
    )
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)
    pd.DataFrame([
        {"column": col, "median": round(float(medians[col]), 4)}
        for col in ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    ]).to_csv("models/impute_values.csv", index=False)

    reloaded = joblib.load("models/delay_model.pkl")
    p_memory = pipeline.predict_proba(X_test)[:, 1]
    p_reload = reloaded.predict_proba(X_test)[:, 1]
    pd.DataFrame([
        {"metric": "test_rows_scored", "value": len(X_test)},
        {"metric": "reload_max_abs_probability_diff", "value": round(float(np.max(np.abs(p_memory - p_reload))), 10)},
        {"metric": "threshold", "value": round(float(config["threshold"]), 4)},
    ]).to_csv("outputs/part_d_reload_check.csv", index=False)


if __name__ == "__main__":
    main()
