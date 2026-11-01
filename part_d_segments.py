"""Part D: hub operating profiles, K-Means segmentation, and tier mismatch evidence."""

import itertools
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def across_within_ratio(X, labels):
    within = []
    across = []
    for i, j in itertools.combinations(range(len(X)), 2):
        distance = float(np.linalg.norm(X[i] - X[j]))
        (within if labels[i] == labels[j] else across).append(distance)
    return float(np.mean(across) / np.mean(within))


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")

    profile = (
        df.groupby("hub_id", sort=True)
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .reset_index()
        .sort_values("hub_id")
        .reset_index(drop=True)
    )

    profile_out = profile.copy()
    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        profile_out[col] = profile_out[col].round(4)
    profile_out.to_csv("outputs/hub_profile.csv", index=False)

    cluster_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    X_scaled = StandardScaler().fit_transform(profile[cluster_cols])

    separation_rows = []
    labels_by_k = {}
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        labels_by_k[k] = labels
        sizes = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
        separation_rows.append({
            "K": k,
            "across_over_within": round(across_within_ratio(X_scaled, labels), 4),
            "cluster_sizes": "-".join(str(int(v)) for v in sizes),
        })
    pd.DataFrame(separation_rows).to_csv("outputs/part_d_separation.csv", index=False)

    labels = labels_by_k[3]
    profile_labeled = profile.copy()
    profile_labeled["cluster"] = labels
    cluster_delay = profile_labeled.groupby("cluster")["delay_rate"].mean().sort_values()
    ordered_clusters = cluster_delay.index.tolist()
    name_map = {
        ordered_clusters[0]: "Standard",
        ordered_clusters[1]: "Watch",
        ordered_clusters[2]: "Escalate",
    }
    profile_labeled["segment"] = profile_labeled["cluster"].map(name_map)

    hub_meta = df[["hub_id", "hub_city", "hub_tier"]].drop_duplicates("hub_id")
    hub_segments = (
        profile_labeled[["hub_id", "segment"]]
        .merge(hub_meta, on="hub_id", how="left", validate="one_to_one")
        .sort_values("hub_id")
        .reset_index(drop=True)
    )
    hub_segments.to_csv("outputs/hub_segments.csv", index=False)

    segment_order = pd.CategoricalDtype(["Standard", "Watch", "Escalate"], ordered=True)
    profile_labeled["segment"] = profile_labeled["segment"].astype(segment_order)
    segment_profile = (
        profile_labeled.groupby("segment", observed=True, sort=True)
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
        .reset_index()
    )
    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        segment_profile[col] = segment_profile[col].round(4)
    segment_profile.to_csv("outputs/segment_profile.csv", index=False)

    expected = {1: "Standard", 2: "Watch", 3: "Escalate"}
    mismatch = hub_segments[
        hub_segments.apply(lambda r: r["segment"] != expected[int(r["hub_tier"])], axis=1)
    ].copy()
    mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)


if __name__ == "__main__":
    main()
