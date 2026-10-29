# Evaluation protocol

CogniShift predicts elevated order-modification activity in a constructed provider-date window. It does not measure clinical fatigue, medical error, burnout, or patient harm. The reported numbers describe a retrospective record-sequence classification experiment.

## Population and windows

The verified local POE extract has 45,154 rows. Removing 792 rows with missing provider identifiers retains 44,362 orders; no timestamps in that extract fail parsing. Only four input columns are read. Missing order types become `Unknown`; transaction names are stripped and uppercased. `CHANGE` and `D/C` count as modifications.

[Data loading](../cognishift/data.py) assigns 07:00–18:59 to day windows and 19:00–06:59 to night windows. Records before 07:00 belong to the previous calendar date. A key combines provider, anchor date, and day/night. This is an operational grouping, not an observed staffing shift.

[Feature construction](../cognishift/features.py) stably sorts each window by timestamp, retaining CSV order for equal timestamps. Windows need at least 15 orders. Exactly orders 1–5 determine predictors and orders 6–15 determine the outcome rate; later records are ignored. The resulting sample has 789 windows from 478 providers. The first recorded order is the window’s observed start, not a clock-in time.

### Chronology audit

604 eligible windows (76.6%) have equal timestamps at the fifth/sixth-record boundary. Each eligible window contains one distinct patient identifier in the local extract. These aggregate checks were performed on the locally held full table; patient identifiers are not read by the evaluation or exported as published evidence.

The windows separate **records**, but do not establish a strictly later outcome timestamp, prospective warning interval, or complete provider workload. The source was described by the project as MIMIC-IV; its [official methods](https://physionet.org/content/mimiciv/3.1/) specify independently shifted dates for each patient. Consequently cross-patient provider chronology and true staffing shifts cannot be inferred from these data without documented extraction provenance. No protocol change was made to hide this limitation. A tie-excluding or patient-aware analysis would be a separately labeled experiment with a different population.

## Predictors

All 12 predictors use only the first five records:

| Feature | Definition |
| --- | --- |
| `early_changes` | Number of `CHANGE` transactions |
| `early_discontinuations` | Number of `D/C` transactions |
| `early_modification_rate` | Fraction with either modification type |
| `early_order_types` | Number of distinct order types |
| `early_order_type_entropy` | Shannon entropy of order types, in bits |
| `early_task_switch_rate` | Fraction of the four adjacent order-type pairs that differ |
| `early_median_gap_minutes` | Median strictly positive adjacent gap, in minutes; missing if none |
| `early_orders_per_hour` | Five / elapsed hours, with a 0.25-hour floor and a 60/hour cap |
| `early_medication_share` | Fraction of order types containing `med`, case insensitive |
| `start_hour_sin`, `start_hour_cos` | Cyclical encoding of first-order hour and minute over 24 hours |
| `start_weekend` | First-order calendar date is Saturday or Sunday |

Provider/shift identifiers and subsequent modification rates are excluded from the predictor list. The rate floor and cap are preserved research assumptions, not calibrated estimates of staffing intensity.

## Fold-local evaluation

[Evaluation code](../cognishift/evaluation.py) uses `GroupKFold(n_splits=5)` without shuffling, grouped by provider ID. Every provider’s examples occur in exactly one validation fold and never in that fold’s training data. Provider-held-out evaluation does not imply patient-held-out evaluation.

For each fold:

1. Compute `q = quantile(training outcome rates, 0.75)` using pandas’ default linear interpolation.
2. Define training and validation labels separately as `outcome_rate > q`, applying the same training-derived `q` to both. Equality is negative. The discrete next-ten rate means prevalence need not be 25%.
3. Instantiate fresh models; fit imputation and LR scaling only on training predictors, then transform held-out predictors with those fitted objects.
4. Evaluate class-1 scores using ROC-AUC. At a fixed score threshold of 0.5, calculate positive-class F1, precision, and recall with zero divisions set to zero.

| Model | Preserved settings |
| --- | --- |
| Logistic regression | Median imputation, keep empty features; standardization; balanced class weights; max 2,000 iterations; seed 42 |
| Random forest | Median imputation, keep empty features; 250 trees; depth 5; minimum leaf size 5; balanced subsample weights; seed 42; parallel fitting |
| Majority baseline | `DummyClassifier(strategy="prior")`; constant training-prior score; class decisions at 0.5 |

Imputation and scaling follow scikit-learn’s [training-only preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). Fewer than five providers or a single training class is an error. A single validation class produces undefined (`NaN`) AUC and a warning; other metrics remain available. Means and sample SD omit undefined AUCs, so the number of valid folds matters. All five AUCs are defined in the published run.

## Reading the results

[Published fold metrics](results/fold_metrics.csv) retain population sizes, training cutoffs, class prevalence, AUC, F1, precision, and recall. Each model has one metric row per fold and one local prediction per eligible window. Summary metrics weight folds equally rather than pooling predictions, particularly because fold cutoffs can differ. Fold SD is neither a confidence interval nor a significance test.

All five published cutoffs are 0.30. Validation positive prevalence is 15.2%–20.9%; training prevalence is 18.2%–19.7%. Logistic regression has mean AUC 0.865137 and random forest 0.862494; random forest has higher mean F1 (0.598249 versus 0.569104). No statistical superiority, hyperparameter search, nested model selection, or independent confirmation is claimed. The baseline supplies a class-imbalance/chance floor; it does not establish the incremental value of all features over early modification activity alone.

Scores from balanced classifiers are not established calibrated probabilities. The dataset, inferred window construction, possible correlated patient records, timestamp ties, and future-availability selection all limit transfer to operational settings.

## Reproduction and artifacts

The CLI uses fixed windows and seed 42. [Run metadata](results/run_summary.json) records aggregate counts, an input digest, parameters, and installed versions without paths or row identifiers. The original and refactored evaluators were compared on the same input: counts, labels, and metrics are unchanged, with numerical prediction equivalence checked within floating-point tolerance. Synthetic runs verify execution but do not reproduce clinical performance.

Input-validation corrections retain mixed-format valid timestamps, exclude blank provider IDs, and preserve numeric-looking provider identifiers as text rather than merging `001` with `1`. These changes have regression coverage and do not change the verified local population or results. Empty cleaned inputs now fail explicitly.

Raw inputs and per-window predictions stay local under ignored directories. Published evidence contains only aggregate fold statistics, a model-comparison chart, and run metadata. There is no trained deployment model, inference service, or clinical alerting system.
