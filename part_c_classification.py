"""Part C: delay classification, feature signal, tuning, and alert-threshold selection."""

import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS


def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def split_data(df):
    X = df[FEATURES]
    y = df["is_delayed"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    model_rows = []
    blind = np.zeros(len(y_val), dtype=int)
    model_rows.append({"model": "blind_all_on_time", **classification_metrics(y_val, blind)})

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ])
    logistic.fit(X_train, y_train)
    model_rows.append({"model": "logistic_regression", **classification_metrics(y_val, logistic.predict(X_val))})

    tree_candidates = []
    for depth in [2, 3, 4, 5, 6, 8, 10]:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        pred = tree.predict(X_val)
        tree_candidates.append((depth, f1_score(y_val, pred), tree, pred))
    best_depth, _, best_tree, best_tree_pred = max(tree_candidates, key=lambda x: (x[1], -x[0]))
    model_rows.append({"model": f"decision_tree_depth_{best_depth}", **classification_metrics(y_val, best_tree_pred)})

    default_forest_all = RandomForestClassifier(n_estimators=100, random_state=42)
    default_forest_all.fit(X_train, y_train)
    model_rows.append({"model": "random_forest_default", **classification_metrics(y_val, default_forest_all.predict(X_val))})
    pd.DataFrame(model_rows).to_csv("outputs/part_c_models.csv", index=False)

    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_
    importances = default_forest_all.feature_importances_
    score_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": f_scores,
        "rf_importance": importances,
    })
    score_df["f_rank"] = score_df["f_score"].rank(method="min", ascending=False).astype(int)
    score_df["is_noise"] = score_df["feature"].isin(NOISE_COLUMNS)
    score_df = score_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    score_df["f_score"] = score_df["f_score"].round(4)
    score_df["rf_importance"] = score_df["rf_importance"].round(4)
    score_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

    select_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        select_rows.append({"k": k, "val_f1": round(float(f1_score(y_val, pipe.predict(X_val))), 4)})
    pd.DataFrame(select_rows).to_csv("outputs/part_c_selectk.csv", index=False)

    selector = SelectKBest(f_classif, k=7)
    X_train_7 = selector.fit_transform(X_train, y_train)
    X_val_7 = selector.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
    grid.fit(X_train_7, y_train)
    grid_pred = grid.best_estimator_.predict(X_val_7)

    randomized = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=9,
        cv=5,
        random_state=42,
    )
    randomized.fit(X_train_7, y_train)
    random_pred = randomized.best_estimator_.predict(X_val_7)

    default7 = RandomForestClassifier(n_estimators=100, random_state=42)
    default7.fit(X_train_7, y_train)
    default_pred = default7.predict(X_val_7)

    tuning_rows = [
        {
            "search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135,
            "best_params": json.dumps(grid.best_params_, sort_keys=True),
            "best_cv_accuracy": round(float(grid.best_score_), 4),
            "val_f1": round(float(f1_score(y_val, grid_pred)), 4),
        },
        {
            "search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45,
            "best_params": json.dumps(randomized.best_params_, sort_keys=True),
            "best_cv_accuracy": round(float(randomized.best_score_), 4),
            "val_f1": round(float(f1_score(y_val, random_pred)), 4),
        },
        {
            "search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
            "best_params": json.dumps({"max_depth": None, "min_samples_leaf": 1, "n_estimators": 100}, sort_keys=True),
            "best_cv_accuracy": np.nan,
            "val_f1": round(float(f1_score(y_val, default_pred)), 4),
        },
    ]
    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

    preference = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
    chosen = sorted(tuning_rows, key=lambda r: (-r["val_f1"], preference[r["search"]]))[0]
    if chosen["search"] == "grid_search":
        params = dict(grid.best_params_)
    elif chosen["search"] == "randomized_search":
        params = dict(randomized.best_params_)
    else:
        params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, **params)),
    ])
    final_model.fit(X_train, y_train)
    val_prob = final_model.predict_proba(X_val)[:, 1]

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_rows = []
    for t in thresholds:
        pred = (val_prob >= t).astype(int)
        threshold_rows.append({"threshold": t, **classification_metrics(y_val, pred)})
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv("outputs/part_c_thresholds.csv", index=False)

    eligible = threshold_df.loc[threshold_df["recall"] >= 0.80, "threshold"]
    chosen_threshold = float(eligible.max()) if len(eligible) else 0.1

    # The test set is touched only here, after configuration and threshold are frozen.
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= chosen_threshold).astype(int)
    pd.DataFrame([{"threshold": chosen_threshold, **classification_metrics(y_test, test_pred)}]).to_csv(
        "outputs/part_c_final_test.csv", index=False
    )

    config = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": int(params.get("n_estimators", 100)),
        "max_depth": "unlimited" if params.get("max_depth") is None else int(params["max_depth"]),
        "min_samples_leaf": int(params.get("min_samples_leaf", 1)),
        "threshold": chosen_threshold,
        "chosen_from": chosen["search"],
    }])
    config.to_csv("outputs/part_c_final_config.csv", index=False)


if __name__ == "__main__":
    main()
