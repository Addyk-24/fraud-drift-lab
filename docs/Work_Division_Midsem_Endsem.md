# Work Division: MLOps-Integrated Fraud Detection with Drift-Aware Retraining

Team of 4 · Guide: Dr. Kapil Vhatkar · Two reviews: Mid-Sem and End-Sem

Replace "Member 2/3/4" with real names. Aditya Katkar is the Team Lead.

## 1. Overall split

| Person | Role | Mid-Sem share | End-Sem share | Total share |
|---|---|---|---|---|
| **Aditya** | Team Lead, core ML + drift + retraining + MLOps | 18 pts (40% of mid) | 25 pts | **43%** |
| Member 2 | Data engineer | 9 pts | 10 pts | 19% |
| Member 3 | EDA and visualization | 9 pts | 10 pts | 19% |
| Member 4 | Preprocessing, imbalance, docs and literature | 9 pts | 10 pts | 19% |
| **Total** | | **45 pts (45% of project)** | **55 pts** | **100%** |

Mid-Sem covers roughly 45% of the project: everything up to "baseline model, chronological simulation and first evidence of drift". The novel parts (retraining policies, cost analysis, SHAP, MLOps) stay for End-Sem.

## 2. Mid-Sem Review (about 45%)

**Goal for the review:** show a clean chronological pipeline, EDA proving that fraud patterns change over time, and a baseline model whose performance decays when it is never retrained.

### Aditya (Lead): 18 pts, the complex work
1. **Project setup:** GitHub repo, folder structure, shared config (window size, random seed, metrics) and branching rules, so everyone's work plugs into one pipeline.
2. **Chronological simulation framework:** sort by `TransactionDT`, split into sequential time windows (for example weekly or monthly over about 6 months), and build the train-on-past / test-on-next-window loop. All End-Sem strategies reuse this loop.
3. **Baseline fraud model:** LightGBM or XGBoost (plus a quick CatBoost comparison) with a time-based split. Metrics: PR-AUC, ROC-AUC, recall at a fixed precision, F1.
4. **"No retraining" strategy and performance-decay curve:** train once on the first window, then score every later window. This is the main Mid-Sem result and the motivation for the project.
5. **Preliminary drift check:** run ADWIN/DDM (River) on the baseline's error stream to get a first look at when drift shows up. Full tuning is End-Sem work.
6. Review the teammates' work and merge it. Lead the Mid-Sem presentation and demo.

### Member 2: Data engineer (9 pts)
1. Download IEEE-CIS from Kaggle and document how the data is obtained.
2. Load `train_transaction` and `train_identity`, left-join on `TransactionID`, and reduce memory with dtype downcasting.
3. Data-quality report: shape, dtypes, duplicates, missing-value percentage per column, inconsistent values.
4. Missing-value handling (drop very sparse columns, impute the rest by feature type) and outlier checks.
5. Save the cleaned dataset as Parquet for the team.

### Member 3: EDA and visualization (9 pts)
1. Descriptive statistics and class-imbalance analysis (fraud rate is about 3.5%).
2. **Fraud rate over time**, transaction volume over time, and fraud by `ProductCD`, card type, email domain and device.
3. **Feature-shift analysis across windows:** KS test or PSI for the top features. This is the evidence that drift exists, and it feeds Aditya's drift section.
4. Correlation analysis and a clean set of plots (Matplotlib/Plotly) for the slides and report.
5. EDA notebook with written insights.

### Member 4: Preprocessing, imbalance, docs and literature (9 pts)
1. Encode categoricals (label/frequency/target encoding, time-safe to avoid leakage) and scale where needed.
2. Feature selection: remove redundant and highly correlated columns (the V columns have many) and rank by importance.
3. Class-imbalance handling: class weights and SMOTE, compared on the baseline. SMOTE is applied inside the training window only.
4. Literature review of the 5 references (Dal Pozzolo 2018, ADWIN, DDM and others) with a short comparison table.
5. Mid-Sem report draft and slide deck assembly (intro, problem, dataset, methodology).

### Mid-Sem deliverables
- Dataset / data collection, cleaned dataset
- EDA notebook and visualizations
- Preprocessing pipeline
- Baseline ML model and no-retraining decay results
- Preliminary drift evidence
- Mid-Sem report and presentation

## 3. End-Sem Review (remaining about 55%)

### Aditya (Lead): 25 pts
1. **Full drift detection:** ADWIN and DDM (optionally Page-Hinkley) tuned on the error stream, plus data-drift detection (PSI/KS) as a second trigger. Compare detection delay and false alarms.
2. **Three retraining strategies in the simulator:** no retraining, fixed-interval (weekly/monthly/quarterly) and drift-triggered retraining, with a sliding window versus an expanding window for the training data.
3. **Cost model:** retraining cost (compute time, number of retrains) and fraud-loss cost (missed fraud amount using `TransactionAmt` versus false-alarm cost). Compare strategies on performance versus cost.
4. **MLOps pipeline:** MLflow experiment tracking and model registry, Dockerized pipeline, GitHub Actions CI, and an automatic retrain trigger wired to the drift detector.
5. Final analysis, conclusions, and integration of everything into the final repo. Lead the final presentation.

### Member 2: 10 pts
1. Productionize the preprocessing into reusable scripts/modules (sklearn pipeline) so the retraining loop can call it automatically.
2. Help set up Docker and the MLflow logging with Aditya.
3. Unit tests for the data pipeline and GitHub Actions checks (lint and tests).
4. Reproducibility documentation (README, environment file, run instructions).

### Member 3: 10 pts
1. **SHAP explainability:** global importance, per-window importance, and how the top features change before and after drift.
2. Visualizations of performance versus time for all three strategies, drift-detection markers on the timeline, and a cost-versus-performance chart.
3. **Dashboard** (Streamlit or Plotly) showing the performance timeline, drift alerts and strategy comparison.

### Member 4: 10 pts
1. Hyperparameter tuning of the final model (Optuna or a grid), run under Aditya's simulator.
2. Sensitivity study: how window size and retrain interval affect results.
3. Final project report, full references, and the final slide deck with the demo script.
4. Optional paper draft if the team targets a conference publication.

### End-Sem deliverables
- Drift detection results, three-strategy comparison and cost analysis
- SHAP explainability results
- Dashboard and working MLOps pipeline (MLflow, Docker, CI)
- Final report, presentation and demonstration, open-source GitHub repo

## 4. Suggested timeline

| Phase | Work | Owner |
|---|---|---|
| Week 1 | Repo setup, data download, merge, cleaning starts | Aditya, Member 2 |
| Week 2 | EDA, encoding, feature selection | Members 3, 4 |
| Week 2-3 | Simulation loop, baseline model | Aditya |
| Week 3-4 | No-retraining decay curve, preliminary drift check, feature-shift plots | Aditya, Member 3 |
| Week 4 | Merge, slides, report, rehearsal | Everyone, Aditya leads |

Adjust the week counts to your real review date.

## 5. Notes and risks

- **Labels:** in IEEE-CIS only `train_*` files have `isFraud`; the Kaggle test set has no labels. Do all chronological splitting inside the train files (about 590k rows).
- **`TransactionDT` is a time offset in seconds, not a date.** It spans about 6 months, so weekly or monthly windows are realistic. Agree on the window size in week 1, because everyone depends on it.
- **Leakage:** fit encoders, scalers and SMOTE on past data only, never on future windows.
- **Hand-offs:** Member 2's Parquet file is the single source for everyone. Member 4's pipeline and Member 3's shift analysis feed directly into Aditya's simulation loop, so agree on the interfaces early.
- **Fairness and workload:** this gives Aditya about 43% of the total (lead and most complex parts) and keeps the other three even at 19% each. If you want a different ratio, change the points in Section 1.

## 6. Research-paper plan (added)

The project is now run as a paper. Every task above produces a figure, table or section of `paper/main.tex`; `README.md` (Results section) is regenerated automatically by `python -m src.make_readme`, so numbers are never copied by hand.

| Paper section | Owner | Needs from the project | Phase |
|---|---|---|---|
| Abstract, Introduction (RQs, contributions), Discussion, Conclusion | Aditya | all results | Mid-Sem draft, End-Sem final |
| Related Work + "gap" paragraph | Member 4 | literature table (at least 8-10 papers, not only the 5 in the proposal) | Mid-Sem |
| Data and Problem Formulation, Table 1 | Member 2 | `data_quality.json` | Mid-Sem |
| EDA paragraph and figures | Member 3 | `eda_*.png`, `eda_stats.json` | Mid-Sem |
| Preprocessing, Model, Experimental Setup | Member 4 | `features.py`, `imbalance_ablation.csv` | Mid-Sem |
| Simulation, drift detection, policies, cost model, results RQ1-RQ3 | Aditya | `baseline_summary.json`, `drift_summary.json`, policy runs | RQ1-2 Mid-Sem, RQ3 End-Sem |
| Explainability (RQ4) | Member 3 | SHAP | End-Sem |
| Reproducibility, code/data availability, threats to validity | Member 2 | repo, Docker, CI | End-Sem |

Publication target: a Scopus-indexed conference (the proposal lists this as an expected outcome). Before submitting, the paper needs the drift-triggered policy, the cost model with a sensitivity analysis, significance tests across windows, and at least one extra experiment (for example label delay of 0/7/14 days) so the contribution is more than "we ran the standard methods".

### Immediate next steps (in order)
1. Everyone: get Kaggle access and run `python -m src.data_download` (or unzip the two train CSVs into `data/raw/`).
2. Aditya: `python run_midsem.py` on the real data, then read the generated README and check that the numbers make sense (see "Risks" below).
3. Member 2 / 3 / 4: read the output of your own module, tune it, and write your paper section from the generated tables.
4. Aditya: build the Mid-Sem deck from README figures 1 to 5.

### Risks to check on the first real run
- The window size (7 days) and the 8-window initial training period are design choices. If a window has very few frauds (under about 30), metrics get noisy; switch to 14-day windows in `config.py`.
- `src/drift_analysis.py` uses a simple degradation-onset rule. If real PR-AUC is noisy it may report no onset or a wrong one; check `decay_curve.png` by eye.
- SMOTE on many categorical columns is crude (integer-coded); say so in the paper instead of over-claiming.
