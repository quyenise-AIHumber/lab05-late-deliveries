Late Deliveries
QuickBite predicts, when a customer places an order, whether delivery is likely to be late. This project builds a reproducible, leak-free scikit-learn pipeline and selects a voucher-offer threshold by minimizing the company's stated cost: $4 for every flagged order, plus $15 for every late order not flagged.

The data is synthetic (make_orders(n=3000, seed=7)). The final model is Logistic Regression, with a decision threshold of 0.25, selected without using the held-out test set.
Setup
Requires Conda and Git. Create the environment using the repository's environment.yml, then install the pinned dependencies and editable package:
conda env create -f environment.yml
conda activate lab05-late-deliveries
python -m pip install -r requirements.txt
python -m pip install -e .
The environment uses Python 3.11 and the conda-forge channel. The model metadata records scikit-learn 1.9.1, the version in which the submitted model was generated. Load joblib files only from trusted sources.
Run
From the repository root, regenerate model-selection reports, evaluate the held-out test set, and save the fitted pipeline and results:
python -m latedeliveries.train
Outputs:
- reports/cv_results.csv: five-fold cross-validation, with mean and standard deviation for five metrics.
- reports/model_cost_comparison.csv: out-of-fold training costs for the three real candidates.
- reports/threshold_costs.csv: tuned model's out-of-fold cost at thresholds 0.05 to 0.95.
- reports/metrics.json: one held-out test evaluation and the three policy costs.
- models/late_orders.joblib: complete fitted pipeline (preprocessing plus classifier).
- models/model_info.json: library version, seed, threshold, feature definitions, selected settings, and test results.
notebooks/analysis.ipynb provides a saved-output analysis of prevalence, comparison results, threshold costs, the test confusion matrix, the three policies, and the final recommendation.
Check it
python -m ruff format --check .
python -m ruff check .
python -m pytest
Locally verified: Ruff format check passed, Ruff lint passed, and 13 tests passed. These are results from the original development environment; verify them again after a fresh clone.
Decisions
Prediction timing and feature selection
The prediction must be actionable at order placement, so only information known at that time is eligible. Features are defined once in pipeline.py:
Type	Included features	Reason
Numeric	distance_km, items, hour	Delivery distance, order size, and placement hour are available at ordering.
Categorical	weather, day_type	Current weather and weekday/weekend are available at ordering.


Excluded variables and the justification for each:
Variable	Why excluded
order_id	Identifier, not a relevant operational driver; using it risks learning order-specific artifacts.
prep_minutes	Actual kitchen preparation duration is observed after the order is placed; using it would look into the future.
customer_rating	Post-delivery customer feedback; directly leaks information about the outcome.
vehicle	Courier/vehicle assignment is not confirmed to occur before the model decision; conservatively excluded.
courier_months	Courier identity/experience is not confirmed to be known at order placement; conservatively excluded.
late	Target variable; it cannot be used as a predictor.


The source generator src/latedeliveries/data.py is preserved unchanged. There are 3,000 orders, including 468 late orders (15.6%); prep_minutes has 189 missing values and customer_rating has 1,489 missing values. Neither post-order field is included in modeling.
Split, preprocessing, and leakage prevention
- A single stratified 75%/25% split uses random_state=42: 2,250 training and 750 held-out test orders.
- All model comparison, hyperparameter selection, and threshold selection use training data only. Stratified five-fold CV uses the same fixed seed.
- The common scikit-learn Pipeline uses a ColumnTransformer: median imputation then standardization for numeric fields; one-hot encoding (handle_unknown="ignore") for categorical fields; followed by the candidate model.
- Preprocessing is fitted within each training fold, preventing validation-fold statistics from leaking into training. The final pipeline is fitted to the complete training subset only after choices are frozen.
- Tests cover a missing eligible numeric value (distance_km), an unseen weather category (hail), and an input containing missing prep_minutes. As prep_minutes is intentionally excluded, that last field is ignored rather than imputed; the missing distance_km verifies numeric imputation.
Honest baseline and candidate comparison
A DummyClassifier(strategy="prior") baseline and three real candidates were evaluated with stratified five-fold training CV. All numbers below are mean ± standard deviation across folds. These are default-threshold classification metrics (for the relevant classifiers); the business-cost comparison below separately optimizes each candidate's decision threshold.
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy | 0.8440 ± 0.0009 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| Logistic Regression | 0.8551 ± 0.0139 | 0.6074 ± 0.1618 | 0.1993 ± 0.0581 | 0.2981 ± 0.0826 | **0.7685 ± 0.0282** |
| Balanced Decision Tree | 0.6987 ± 0.0644 | 0.3016 ± 0.0704 | 0.6181 ± 0.0562 | 0.3968 ± 0.0427 | 0.7249 ± 0.0237 |
| Random Forest | 0.8569 ± 0.0112 | 0.6420 ± 0.1354 | 0.1651 ± 0.0624 | 0.2608 ± 0.0908 | 0.7577 ± 0.0254 |


Accuracy alone is misleading for a late-delivery rate of just 15.6%: the Dummy baseline achieves 84.40% accuracy while identifying no late orders. Recall matters because missed delays incur $15, but a model that flags too many on-time deliveries also wastes $4 vouchers. Therefore the final candidate decision is based on out-of-fold expected business cost, not a single classification metric.
Real candidate	Lowest training out-of-fold cost	Chosen threshold for comparison
Logistic Regression	$4,301	0.25
Random Forest	$4,337	0.25
Balanced Decision Tree	$4,479	0.70


Choice: Logistic Regression had the lowest measured out-of-fold training cost, beating Random Forest by $36 and the balanced tree by $178 on the same 2,250 training orders. Its ROC-AUC was also highest among the real candidates. These differences are modest and should not be interpreted as proof of future superiority.
Hyperparameter tuning
GridSearchCV tuned the chosen full pipeline using five-fold stratified CV, scoring="roc_auc", and six combinations from two parameters:
- model__C: 0.1, 1.0, 10.0.
- model__class_weight: None, "balanced".
ROC-AUC assesses probability ranking without locking in a classification threshold. Operational threshold selection is performed separately using QuickBite's voucher/refund costs.
Comparison	ROC-AUC
Untuned CV mean	0.7685
Untuned fold-to-fold standard deviation	0.0282
Tuned CV mean	0.7686
Difference in reported means	+0.0001


Best settings: model__C=0.1, model__class_weight=None.

Did tuning beat the fold-to-fold spread? No. The ~0.0001 improvement is far below the 0.0282 untuned fold-to-fold standard deviation; it is not evidence of a material improvement. The tuned pipeline is kept because its settings and subsequent threshold were selected from training data, not from test results.
Cost-based threshold selection
The cost function is implemented in evaluation.py:

Total cost (USD) = 4 × (TP + FP) + 15 × FN.

A flagged order always costs $4, whether it arrives late or on time; an unflagged late order costs $15. After tuning, cross_val_predict(..., method="predict_proba") generated out-of-fold probabilities on the training set only. Thresholds were evaluated from 0.05 to 0.95 in 0.05 increments, with the lowest-cost threshold chosen. The test set played no role in this choice.
Training-only policy	Total cost for 2,250 orders
Flag nobody	$5,265
Flag everybody	$9,000
Tuned model at threshold 0.25	$4,287


Selected threshold: 0.25. The business may rationally flag an order below the default 0.50 cutoff because a missed delay ($15) is more expensive than a voucher ($4). A purely theoretical cost break-even probability of $4/$15 ≈ 0.267 assumes well-calibrated probabilities; the empirical search selected 0.25 on the required 0.05 grid. The tuned model's training cost was $14 lower than the untuned model's $4,301; that small difference alone does not establish a dependable improvement.
Leakage sensitivity check (training folds only)
As a diagnostic, add only the excluded customer_rating column to a separate Logistic Regression pipeline and repeat stratified five-fold ROC-AUC cross-validation on the training subset. Do not change the production feature lists or examine the held-out test set. This comparison was independently reproduced from the supplied seeded data generator using the same split and folds:
Training-only check	Mean ROC-AUC
Legitimate order-placement features	0.7685
Illegitimately adding customer_rating	0.9038
Artificial apparent improvement	+0.1352


This is a data-leakage warning, not an improvement in deployable performance. Customer ratings are collected after deliveries, and the synthetic generator makes them depend on late. The leaky model is never used for selection, final evaluation, or saving. These diagnostic figures should be confirmed by executing the notebook in the submitted environment.
Result
All modeling decisions above were made before the final pipeline was fitted on the 2,250 training orders and scored against the 750 held-out test orders. The following are the saved, single held-out evaluation results from reports/metrics.json.
Confusion matrix
Rows are actual outcomes; columns are predicted outcomes using threshold 0.25.
Actual / Predicted	On time (0)	Late (1)
On time (0)	TN = 567	FP = 66
Late (1)	FN = 64	TP = 53


- 119 orders flagged: 53 true late orders plus 66 on-time orders.
- 64 late orders missed: these generate the larger $15 penalty.
- 117 truly late orders appear in the held-out set.
Held-out classification metrics
Metric	Test result
Precision	0.4454 (44.54%)
Recall	0.4530 (45.30%)
F1	0.4492
ROC-AUC	0.7646


Comparison of voucher policies
Policy	Test-set cost	Explanation
Flag nobody (current policy)	$1,755	117 missed delays × $15
Flag everybody	$3,000	750 vouchers × $4
Model at threshold 0.25	$1,436	119 vouchers × $4 + 64 missed delays × $15


Measured savings versus flag nobody: $319 on 750 simulated test orders, or 18.18%. The model is also $1,564 cheaper than flagging everybody in this test. The savings are modest in absolute dollars, while recall is only 45.30%; more than half of actual delays were not flagged.

Recommendation: Favor a limited pilot and continued validation, not automatic full deployment. The cost-based model improved outcomes on this one synthetic held-out sample, but that sample alone does not establish a reliable production cost reduction. Validate feature availability, probability calibration, real voucher effectiveness, operational constraints, and drift using real order data before relying on the policy.

#What the tests cover
The 13 passing pytest cases cover:
1. Parametrized exclusion tests for order_id, prep_minutes, customer_rating, late, vehicle, and courier_months (6 cases).
2. Stratified and reproducible splitting with the fixed seed (1 case).
3. Valid predicted probabilities, bounded between 0 and 1 and summing to 1 (1 case).
4. Prediction with a missing numeric input, an unseen weather category, and a missing excluded field (1 case).
5. Hand-calculated voucher-plus-missed-delay cost (1 case).
6. Threshold selection when flagging nobody is cheapest (1 case).
7. Threshold selection when flagging everybody is cheapest (1 case).
8. Saving and reloading a model using pytest's tmp_path, with matching predictions before and after (1 case).
Not tested: generalization to real customer orders, because only synthetic data is provided. Additional future tests could check feature-schema enforcement and package execution from an independent operating system.

#Known limitations
- Synthetic data: The estimated cost saving is demonstrated only on generated orders, not actual QuickBite operating history; it is not a forecast of financial savings.
- Limited recall: On the held-out set, the model missed 64 of 117 late deliveries. Cost reduction must not be confused with high capture of service failures.
- Feature-timing assumptions: vehicle and courier_months were excluded conservatively because assignment timing is unknown. If operational evidence proves either is available when predictions are made, the features can be reconsidered using a new pre-specified evaluation.
- Probability calibration and drift: Ranked probabilities may require calibration before interpreting them literally as delay risk. Changes in traffic, weather, preparation, courier allocation, and customer behavior can change performance and the best threshold.
- Selection variability: Model and threshold selection reused training-fold evidence; the observed differences between candidate costs and between tuned/untuned performance are small. The held-out test was used once for final assessment and must not be used for further tuning.
- Business rule simplification: The assignment assumes every voucher works and fixed $4/$15 costs; actual customer response and compensation cost can differ.