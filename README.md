# Delhivery Last-Mile Delay Intelligence Pipeline

This capstone builds one connected offline machine-learning pipeline for three operational questions: estimating parcel delivery hours, identifying parcels likely to miss their promised time, and segmenting hubs by observed operating behaviour. The dataset is synthetic; no real Delhivery shipment data is used.

## How to run

From the repository root, use the exact sequence below:

```bash
python -m pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`generate_data.py` should print `24 1224 48`. Part A creates the 1,200-row modelling table and the shared preprocessing is defined only once in `src/shipment_prep.py`. `package_model.py` regenerates `models/delay_model.pkl` from the final Part C configuration. The committed pickle loads reliably only with the scikit-learn version pinned in `requirements.txt`; if an installed version differs, regenerate the model by running `python package_model.py` rather than relying on the committed pickle.

Repository flow:

`generate_data.py` → `part_a_clean_eda.py` → `part_b_regression.py` / `part_c_classification.py` → `part_d_segments.py` → `package_model.py` → `predict_delay.py`.

## Data findings

- **Express is delayed most often.** Its delay rate is **0.4897** (119 delayed among 243 shipments), compared with **0.2056** for Standard and **0.0857** for Economy (`outputs/part_a_delay_by_group.csv`).
- **Delay risk increases strongly with hub tier.** Tier 1 has a delay rate of **0.0400**, Tier 2 **0.2150**, and Tier 3 **0.4400** (`outputs/part_a_delay_by_group.csv`).
- **Distance is operationally important.** Delay rate rises from **0.0219** for 0–200 km, to **0.1368** for 201–400 km, **0.4364** for 401–600 km, and **0.8523** for 601+ km (`outputs/part_a_delay_by_group.csv`).
- The three strongest absolute correlations with `delivery_hours`, excluding the target itself, are **promised_hours = +0.8372**, **distance_km = +0.4013**, and **n_stops = +0.3247** (`outputs/part_a_correlations.csv`). Correlation is descriptive evidence, not proof of causation.
- The filled `declared_value_inr` distribution is right-skewed: mean **INR 1,533.0542** versus median **INR 1,121.5000**. After the natural-log transform, mean **7.0277** and median **7.0224** are much closer, so `log_declared_value` reduces skew and gives the models a more compact representation (`outputs/part_a_checks.csv`).

## Model results (regression)

The mean-only baseline has validation **R² = -0.0212**, meaning it is slightly worse than simply using the validation target mean as a reference and has no useful parcel-level explanatory power.

The validation winner is **linear regression**, with **RMSE = 4.8912 hours** and **R² = 0.9313**. Gradient boosting is extremely close at **RMSE = 4.9165 hours** and **R² = 0.9306** (`outputs/part_b_metrics.csv`). Following the project rule, the candidate with the lowest validation RMSE is the final model; its one-time test score is **R² = 0.9493**.

The tree-depth diagnostic shows the bias-variance trade-off clearly. At depth 1, train RMSE is **11.6334** and validation RMSE is **12.1324**, so both errors are high: this is high bias. The best tree depth is 6 with validation RMSE **6.2796**. With no depth limit, train RMSE falls to **0.0000** while validation RMSE rises to **7.8368**, a large generalization gap indicating high variance (`outputs/part_b_tree_depth.csv`).

For gradient boosting, learning rate **0.1** is best with validation **R² = 0.9306**. Learning rate **0.01** learns too slowly within 100 estimators (**R² = 0.7127**), while **1.0** is too aggressive and gives a poorer **R² = 0.8440** (`outputs/part_b_gb_learning_rate.csv`).

Cross-validation slightly prefers gradient boosting (**mean R² = 0.9353**) over linear regression (**0.9261**), while the fixed validation split slightly prefers linear regression. I would ship linear regression for this capstone because the brief explicitly defines the final model by the lowest validation RMSE, and I would record the cross-validation disagreement as evidence that the two models are very close rather than use the test set to choose between them.

## Model results (classification)

The blind `all on time` baseline reaches **0.7667 accuracy** but has **0.0000 recall**: it misses all 42 delayed validation parcels and is therefore operationally useless (`outputs/part_c_models.csv`).

The five bookkeeping/noise columns together account for **0.2245** of the default forest's importance. All five are outside the top seven by F-score; their ranks are 8, 12, 14, 16 and 17, and each F-score is below 10 (`outputs/part_c_feature_scores.csv`). This shows why feature selection is useful even though a forest can assign some importance to noisy variables.

The `SelectKBest` experiment gives validation F1 values of **0.6988, 0.8333, 0.7857, 0.8049 and 0.7848** for k = 5, 7, 10, 12 and 17 respectively. Therefore the required **k = 7** setting is also the strongest of the tested values on this validation split (`outputs/part_c_selectk.csv`).

Tuning did not beat the untuned forest on the same seven selected features. Validation F1 is **0.7750** for grid search, **0.7848** for randomized search, and **0.8333** for the default forest (`outputs/part_c_tuning.csv`). With only 180 validation rows, small differences can be unstable; this is a reason not to keep increasing tuning complexity after looking at the same validation set.

The operations rule chooses threshold **0.5**, the highest tested threshold with validation recall at least 0.80. At 0.5, validation precision is **0.7955** and recall is **0.8333**; at 0.6 precision improves to **0.8889** but recall falls to **0.7619**, below the operational requirement (`outputs/part_c_thresholds.csv`).

On the untouched test set at threshold 0.5, precision is **0.6889**, recall **0.7561**, and F1 **0.7209** (`outputs/part_c_final_test.csv`). Test recall is lower than validation recall, which should be reported rather than hidden; the test set contains only 41 delayed parcels, so each missed delayed parcel changes recall noticeably.

## Operations recommendation

The **Escalate** segment contains nine hubs: HUB13 Nagpur (**0.4600** delay rate), HUB15 Kochi (**0.3600**), HUB17 Patna (**0.4600**), HUB18 Ranchi (**0.3800**), HUB19 Guwahati (**0.6200**), HUB20 Raipur (**0.4800**), HUB21 Bhubaneswar (**0.4000**), HUB23 Siliguri (**0.4400**) and HUB24 Agartala (**0.5400**) (`outputs/hub_segments.csv` and `outputs/hub_profile.csv`). These hubs should receive the strongest operating playbook because the segment is defined by the highest hub-level delay behaviour.

The tier label should not be treated as a perfect proxy for actual operating behaviour. Four hubs mismatch: **HUB07 Kolkata** is Tier 1 but behaves as Watch; **HUB13 Nagpur** and **HUB15 Kochi** are Tier 2 but behave as Escalate; and **HUB22 Dehradun** is Tier 3 but behaves as Watch (`outputs/tier_vs_segment_mismatch.csv`). The tier remains useful metadata, but the behavioural segment should guide the delay-response playbook.

The alert threshold is **0.5**. On test data, precision **0.6889** means roughly **6.9 of every 10** flagged parcels are actually delayed, while recall **0.7561** means roughly **7.6 of every 10** delayed parcels are caught. On the 48 new parcels, **8 are flagged** and **6 of those require hub-manager escalation** (`outputs/new_shipment_actions.csv`).

K = 3 is kept because operations requires exactly three playbooks and the separation ratio is already strong at **5.3928** with cluster sizes **9-8-7**. Although the ratio increases to **6.3047** at K = 4 and **6.8866** at K = 5, those solutions create clusters smaller than six hubs (sizes **8-7-5-4** and **8-7-4-3-2**), which are too small to justify separate operating playbooks (`outputs/part_d_separation.csv`).

## Production plan

**Production-readiness mapping.** Accuracy is checked through held-out regression and classification metrics. Reliability is supported by a serialized pipeline and `outputs/part_d_reload_check.csv`, where the reload probability difference is **0.0**. Reproducibility comes from pinned package versions, fixed random states, committed synthetic data and deterministic scripts. Performance is reasonable because scoring is a single local pipeline pass over tabular data with no network dependency. Maintainability comes from one shared preprocessing module and separate scripts for each stage. Consistency is enforced because training and prediction both call `clean_shipments()` and `build_features()`. Monitorability is supported by explicit probability, flag, segment and action fields in the scoring output.

**Monitoring triggers.** After outcomes become known, investigate if the weekly delayed share rises above the historical **0.2317** baseline; investigate if measured recall falls below the held-out **0.7561**; and investigate if measured precision falls below **0.6889**. These are initial baseline triggers, not permanent service-level targets, and should be revised after enough real operational history is accumulated.

**Retraining and drift.** Recompute monitoring metrics weekly and retrain on a regular monthly cycle, or earlier when a monitoring trigger is breached and the change persists. The 15–28 September sale period represented by `sale_event_flag` should be compared with similar sale-event periods rather than ordinary weeks. Drift here would include persistent changes in distance mix, hub-level delay rates, route stops, rain/event frequency, service mix, or the relationship between those inputs and observed delays.

## Limitations

1. The dataset is entirely synthetic, so the measured performance demonstrates pipeline behaviour rather than real-world Delhivery performance.
2. The final classification test set contains only 41 delayed parcels, so precision and recall have meaningful sampling uncertainty.
3. The alert threshold is selected from one 180-row validation split and may not be optimal after deployment or under a different operational cost structure.
4. Hub segmentation is based on only 50 historical shipments per hub and assumes three operational playbooks because the business requirement fixes K = 3.
5. The model does not include live traffic, detailed weather intensity, rider availability, route topology or downstream operational interventions, all of which could affect real last-mile delay risk.
