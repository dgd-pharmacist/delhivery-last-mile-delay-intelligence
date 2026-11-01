"""Part B: delivery-hours regression model comparison and diagnostics."""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    checks = pd.DataFrame([
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", float(np.abs(X_train_scaled.mean(axis=0)).max())),
        ("test_scaled_max_abs_mean", float(np.abs(X_test_scaled.mean(axis=0)).max())),
    ], columns=["metric", "value"])
    checks["value"] = checks["value"].map(lambda v: round(float(v), 4))
    checks.to_csv("outputs/part_b_checks.csv", index=False)

    depth_values = [1, 2, 3, 4, 6, 8, 12, None]
    depth_rows = []
    for depth in depth_values:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        depth_rows.append({"max_depth": "unlimited" if depth is None else str(depth),
                           "train_rmse": round(float(train_rmse), 4),
                           "val_rmse": round(float(val_rmse), 4)})
    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)
    depth_numeric = [(d, row["val_rmse"]) for d, row in zip(depth_values, depth_rows)]
    best_depth = min(depth_numeric, key=lambda t: (t[1], float("inf") if t[0] is None else t[0]))[0]

    models = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metric_rows = []
    baseline_pred = np.full(len(y_val), y_train.mean())
    metric_rows.append({"model": "mean_baseline", "split": "val", **regression_metrics(y_val, baseline_pred)})

    fitted = {}
    validation_rmse = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        pred = model.predict(X_val)
        m = regression_metrics(y_val, pred)
        validation_rmse[name] = m["RMSE"]
        metric_rows.append({"model": name, "split": "val", **m})

    lr_rows = []
    for lr in [0.01, 0.1, 1.0]:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        lr_rows.append({
            "learning_rate": lr,
            "val_rmse": round(float(np.sqrt(mean_squared_error(y_val, pred))), 4),
            "val_r2": round(float(r2_score(y_val, pred)), 4),
        })
    pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    ranked = sorted(validation_rmse, key=lambda n: validation_rmse[n])
    top_two = ranked[:2]
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name in top_two:
        # Clone-like fresh estimator to avoid any dependence on previous fit state.
        fresh = {
            "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        }[name]
        scores = cross_val_score(fresh, X_tv, y_tv, cv=kfold)
        cv_rows.append({"model": name, "mean_r2": round(float(scores.mean()), 4), "std_r2": round(float(scores.std()), 4)})
    pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

    final_name = ranked[0]
    test_pred = fitted[final_name].predict(X_test)
    metric_rows.append({"model": final_name, "split": "test", **regression_metrics(y_test, test_pred)})
    pd.DataFrame(metric_rows).to_csv("outputs/part_b_metrics.csv", index=False)


if __name__ == "__main__":
    main()
