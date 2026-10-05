# Demo guide (review day)

Start before the review: `streamlit run app/dashboard.py` (opens http://localhost:8501). Keep a backup: screenshots of each page, and the README (it has every figure).
Do NOT re-run the pipeline in front of the professor; it takes minutes. The dashboard only reads the saved `results/` folder.

## 5-minute flow (Aditya presents, teammates answer on their own parts)

| Min | Page | Say | Click |
|---|---|---|---|
| 0:00 | Overview | "A fraud model is trained once and deployed; fraud changes. We replay IEEE-CIS chronologically to measure decay, detect it, and decide when retraining is worth the cost." | Point at the 3 headline cards |
| 0:45 | Data & EDA (Member 3) | "Severe imbalance; fraud rate and category mix move week to week, so the data is not stationary." | Tabs: *Fraud rate over time*, *Fraud pattern shift* |
| 1:30 | Live replay (Aditya) | "Each step is one new week. The model never saw these transactions." | Policy `no_retrain` -> **Play**; point at the metric line and the score histogram (fraud scores overlap more as weeks pass); show the risky-transaction table |
| 2:30 | Live replay | "Same weeks, but retraining every 4 weeks." | Switch policy to `fixed_4w` -> Play; green "retrained" lines |
| 3:15 | Drift (Aditya) | "The data itself is drifting: a classifier can tell later weeks from the training period almost perfectly." | *Adversarial validation* tab, then *PSI* tab, zoom on the most shifted feature |
| 4:00 | Policies & cost | "Retraining is not free, so we compare policies on performance and cost. These sliders are assumptions; the break-even column does not depend on them." | Move the *cost of one retraining* slider and watch the cheapest policy change |
| 4:45 | Explainability (Member 3) | "TreeSHAP shows which signals the deployed model relies on and whether that changes." | Bar chart early vs late |

## Questions the professor is likely to ask, and honest answers

1. **Why is the static split better than the deployment result?** A random split lets the model see the future (same weeks in train and test). Deploy-once is the realistic protocol; the gap is our RQ1 result (Overview page, *Live replay*).
2. **Where could data leakage come from, and how do you prevent it?** `TransactionDT` is not a feature; feature selection is fitted on the training window only; the threshold is tuned on the last training window; a retrain at week *w* sees only weeks < *w*. These are enforced by `tests/test_pipeline.py`.
3. **Why PR-AUC and not accuracy?** With a few percent fraud, accuracy is meaningless; PR-AUC focuses on the rare class. We also report recall at 1% false-positive rate.
4. **The drift detectors fire before performance drops. Are they wrong?** Raw ADWIN/DDM on a never-retrained model's error stream are noisy (Drift page says so). The drift-triggered policy adds increase-only filtering, a cool-down and a reset after each retrain; whether it beats fixed-interval retraining is shown on the *Policies & cost* page, including if it does not.
5. **Where do the cost numbers come from?** They are assumptions (review $2, retrain $500, full loss of missed fraud) that you can change live; the break-even retrain cost removes the dependence on the retrain price, and the sensitivity heat-map shows where the ranking flips.
6. **Is this production MLOps?** The pipeline simulates deployment; the retraining trigger logic, experiment tracking (MLflow), tests, CI and a Docker image exist, but no real streaming service is deployed. Say so.
7. **Do labels arrive instantly in real life?** No, chargebacks take days or weeks. `label_delay` in `config.py` models it; the policies page shows delay 0/7/14 days for fixed retraining.
8. **The features are anonymous (V*, C*, D*).** Correct; SHAP tells us which signals matter and when, not their business meaning. That is a limitation of the dataset.
9. **Single dataset / no calendar dates?** Yes, a limitation: `TransactionDT` is an offset, so seasonality cannot be identified. Listed under threats to validity.

## If something breaks
- Dashboard will not start: `pip install streamlit plotly pyarrow`, then run from the project root.
- Blank page / "No results found": `results/` is missing; run `python run_midsem.py --skip-download`.
- Slow Play: untick nothing; choose `no_retrain` (smaller), or step the slider manually.

## Numbers worth remembering (real run)
- 590,540 transactions, 3.5% fraud, 182 days, 26 weekly windows, trained on first 8.
- Random split PR-AUC 0.72 vs 0.48 deploy-once: a static split overstates by about one third.
- Never retrained: PR-AUC 0.546 -> 0.471 (-13.7%, Spearman -0.49, p=0.039). Degradation is real but moderate and noisy.
- Adversarial AUC 0.77 -> 0.97: the inputs drift far more than the performance does. Most shifted feature: id_31 (browser version).
- Policies: fixed 2-weekly best (0.546, +16%), fixed 4-weekly 0.530, drift-triggered ADWIN 0.534 with 4 retrains, i.e. no better than fixed at the same budget. Say this openly; it is a legitimate finding.
- Retraining pays for itself: break-even cost per retrain is at least $17k against an assumed $500.
- Label delay hurts: fixed 4-weekly PR-AUC 0.530 -> 0.516 (7 days) -> 0.496 (14 days).
- SHAP: same features matter early and late (rank corr 0.995); browser/OS strings lose importance.
