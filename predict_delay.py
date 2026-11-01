"""Score new shipments and convert delay probabilities into operational actions."""

import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments


def main():
    raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    threshold = float(pd.read_csv("models/delay_threshold.csv").iloc[0]["threshold"])
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))
    segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]
    model = joblib.load("models/delay_model.pkl")

    clean, _ = clean_shipments(raw, medians=medians)
    featured = build_features(clean, hubs)
    probability = model.predict_proba(featured[feature_order])[:, 1]

    scored = featured[["shipment_id", "hub_id"]].copy()
    scored["delay_probability"] = probability.round(4)
    scored["flagged"] = (scored["delay_probability"] >= threshold).astype(int)
    scored = scored.merge(segments, on="hub_id", how="left", validate="many_to_one", sort=False)

    def action(row):
        if row["flagged"] == 0:
            return "No action"
        if row["segment"] == "Escalate":
            return "Escalate to hub manager"
        return "Proactive customer call"

    scored["action"] = scored.apply(action, axis=1)
    scored = scored[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    scored.to_csv("outputs/new_shipment_actions.csv", index=False)


if __name__ == "__main__":
    main()
