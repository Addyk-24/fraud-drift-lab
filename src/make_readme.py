import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from src import config

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def load_json(name):
    p = config.RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def load_csv(name):
    p = config.TABLE_DIR / name
    return pd.read_csv(p) if p.exists() else None


def md_table(df, floatfmt="{:.4f}"):
    if df is None or df.empty:
        return "_not generated yet_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def fig(readme_dir, name, caption):
    p = config.FIG_DIR / f"{name}.png"
    if not p.exists():
        return ""
    rel = Path(os.path.relpath(p, readme_dir)).as_posix()
    return f"![{caption}]({rel})\n*{caption}*\n\n"


def build(readme_dir) -> str:
    dq, eda = load_json("data_quality.json"), load_json("eda_stats.json")
    base, drift = load_json("baseline_summary.json"), load_json("drift_summary.json")
    if dq is None:
        return ("> Results have not been generated yet. Put the Kaggle files in `data/raw/` and run "
                "`python run_midsem.py` (see *How to run*).\n")
    s = []
    if dq.get("data_source") == "synthetic":
        s.append("> **WARNING: the numbers below come from SYNTHETIC smoke-test data, not IEEE-CIS. "
                 "Do not report them.**\n")
    s.append(f"_Last generated from `{config.RESULTS_DIR.name}/` - data source: **{dq.get('data_source', '?')}**._\n\n")

    s.append("### 1. Dataset and cleaning\n")
    rows = [
        ("Transactions (rows)", f"{dq['merged_shape'][0]:,}"),
        ("Columns: transaction / identity / merged", f"{dq['transaction_shape'][1]} / {dq['identity_shape'][1]} / {dq['merged_shape'][1]}"),
        ("Transactions with an identity record", f"{dq['identity_coverage_pct']}%"),
        ("Fraud transactions", f"{dq['n_fraud']:,} ({dq['fraud_rate_pct']}%)"),
        ("Time span", f"{dq['time_span_days']} days"),
        ("Overall missing values", f"{dq['overall_missing_pct']}%"),
        ("Columns with >90% missing (dropped)", dq["columns_over_90pct_missing"]),
        ("Constant columns dropped", len(dq.get("dropped_constant_columns", []))),
        ("Duplicate TransactionIDs removed", dq["rows_dropped_duplicate_id"]),
        ("Final shape (rows x cols)", f"{dq['final_shape'][0]:,} x {dq['final_shape'][1]}"),
        ("Memory after downcasting", f"{dq['memory_mb_after_downcast']} MB"),
    ]
    s.append(md_table(pd.DataFrame(rows, columns=["Statistic", "Value"]).astype(str)))

    if eda:
        s.append("\n### 2. Exploratory data analysis\n")
        rows = [("Imbalance ratio (legit : fraud)", f"{eda['imbalance_ratio']} : 1"),
                ("Weekly fraud rate, min - max", f"{eda['weekly_fraud_rate_pct_min']}% - {eda['weekly_fraud_rate_pct_max']}%")]
        for k, lab in (("median_amount_legit", "Median amount, legitimate"), ("median_amount_fraud", "Median amount, fraud"),
                       ("fraud_rate_with_identity_pct", "Fraud rate WITH identity record (%)"),
                       ("fraud_rate_without_identity_pct", "Fraud rate WITHOUT identity record (%)"),
                       ("peak_fraud_hour", "Hour with highest fraud rate (relative)")):
            if k in eda:
                rows.append((lab, eda[k]))
        if "fraud_rate_by_ProductCD_pct" in eda:
            rows.append(("Fraud rate by ProductCD (%)", ", ".join(f"{k}: {v}" for k, v in eda["fraud_rate_by_ProductCD_pct"].items())))
        if "top_correlated_features" in eda:
            rows.append(("Most target-correlated numeric features", ", ".join(f"{k} ({v})" for k, v in eda["top_correlated_features"].items())))
        s.append(md_table(pd.DataFrame(rows, columns=["Statistic", "Value"]).astype(str)))
        s.append("\n")
        for n, c in (("eda_class_balance", "Class imbalance"), ("eda_fraud_over_time", "Volume and fraud rate per week"),
                     ("eda_fraud_by_category", "Fraud rate by category"), ("eda_pattern_shift", "Fraud rate over time by category (pattern shift)"),
                     ("eda_amount", "Transaction amount by class"), ("eda_amount_over_time", "Median amount per week"),
                     ("eda_missingness", "Missingness"), ("eda_hourly", "Daily cycle"),
                     ("eda_target_correlation", "Correlation with target")):
            s.append(fig(readme_dir, n, c))

    if base:
        d = base["data"]
        s.append("### 3. Chronological simulation and baseline (RQ1)\n")
        s.append(f"Windows of **{d['window_days']} days**, **{d['n_windows']}** windows in total; model trained on the first "
                 f"**{d['initial_train_windows']}** windows. Features: {d['n_candidate_features']} candidates -> "
                 f"**{d['n_selected_features']}** after dropping {d['n_dropped_correlated']} highly correlated columns "
                 f"(fitted on the training window only).\n\n")
        nr = base["policies"]["no_retrain"]
        rows = []
        for m, lab in (("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("recall_at_1pct_fpr", "Recall @ 1% FPR"), ("f1", "F1")):
            x = nr["decay"][m]
            rows.append((lab, round(x["first3_mean"], 4), round(x["last3_mean"], 4), f"{x['relative_drop_pct']:.1f}%",
                         f"{x['slope_per_week']:+.4f}", f"{x['spearman_rho']:+.2f} (p={x['spearman_p']:.3g})"))
        s.append("**No retraining - degradation of the deployed model**\n\n")
        s.append(md_table(pd.DataFrame(rows, columns=["Metric", "First 3 weeks", "Last 3 weeks", "Relative drop",
                                                      "Slope / week", "Spearman rho vs time"])))
        s.append("\n")
        if "evaluation_protocols" in base:
            s.append("**Same learner, different evaluation protocol** (why static splits are optimistic)\n\n")
            s.append(md_table(pd.DataFrame(base["evaluation_protocols"])))
            s.append("\n")
        if "model_comparison" in base:
            s.append("**Model comparison (same chronological protocol)**\n\n")
            s.append(md_table(pd.DataFrame(base["model_comparison"])))
            s.append("\n")
        for n, c in (("decay_curve", "Performance of the never-retrained model by deployment week"),
                     ("evaluation_protocols", "Static vs temporal vs deploy-once evaluation"),
                     ("top_features", "Top features of the baseline model"),
                     ("decay_curve_fixed_4w", "Fixed 4-week retraining (preview of End-Sem comparison)")):
            s.append(fig(readme_dir, n, c))
        extra = [k for k in base["policies"] if k != "no_retrain"]
        for k in extra:
            p = base["policies"][k]
            s.append(f"`{k}`: mean PR-AUC {p['mean_pr_auc']:.4f}, {p['run']['n_retrains']} retrains, "
                     f"{p['run']['total_train_seconds']:.0f}s training (no_retrain mean PR-AUC {nr['mean_pr_auc']:.4f}).\n\n")

    ab = load_csv("imbalance_ablation.csv")
    if ab is not None:
        s.append("### 4. Class-imbalance handling\n")
        s.append(md_table(ab))
        s.append("\n" + fig(readme_dir, "imbalance_ablation", "Imbalance ablation"))

    if drift:
        s.append("### 5. Preliminary drift analysis (RQ2)\n")
        s.append(f"Reference degradation onset (PR-AUC >20% below early level for 2 weeks): "
                 f"**week {drift['degradation_onset_window']}**.\n\n")
        rows = [(k, v["n_detections"], v["first_detection_window"], v["delay_vs_onset_weeks"], v["detection_windows"])
                for k, v in drift["detectors"].items()]
        s.append(md_table(pd.DataFrame(rows, columns=["Detector", "# detections", "First detection (week)",
                                                      "Delay vs onset (weeks)", "All detection weeks"])))
        a = drift["adversarial_auc"]
        s.append(f"\nAdversarial AUC (reference vs week): first deployment week {a['first_deployment_week']:.3f}, "
                 f"last {a['last_deployment_week']:.3f}, max {a['max']:.3f}. "
                 f"PSI > 0.25 on {drift['psi']['n_features_psi_over_0_25_in_last_window']} of the top-20 features in the last week. "
                 f"Most shifted features (mean PSI, last 4 weeks): "
                 + ", ".join(f"{k} ({v})" for k, v in drift["psi"]["most_shifted_features"].items()) + ".\n\n")
        for n, c in (("drift_detection", "Detector alarms against measured degradation"),
                     ("psi_heatmap", "PSI of top features over time"),
                     ("adversarial_drift", "Adversarial validation AUC over time")):
            s.append(fig(readme_dir, n, c))
    pol = load_json("policy_summary.json")
    if pol:
        s.append("### 6. Retraining policies and cost (RQ3)\n")
        d = pol["cost_defaults"]
        s.append(f"Cost = fraud_loss_rate x missed fraud amount + review cost x flagged + retrain cost x #retrains. "
                 f"Defaults (assumptions): review ${d['review_cost']}, retrain ${d['retrain_cost']}, loss rate {d['fraud_loss_rate']}. "
                 f"Break-even retrain cost = price per retraining at which the policy stops paying off vs never retraining. "
                 f"Wilcoxon p: paired per-week PR-AUC vs never retraining.\n\n")
        t = pd.DataFrame(pol["policies"])
        cols = ["policy", "label_delay_days", "n_retrains", "mean_pr_auc", "pr_auc_gain_vs_no_retrain_pct",
                "wilcoxon_p_vs_no_retrain", "mean_recall_at_1pct_fpr", "train_seconds", "total_cost",
                "cost_saving_vs_no_retrain_pct", "break_even_retrain_cost"]
        s.append(md_table(t[cols].round(3), floatfmt="{:.3f}"))
        s.append(f"\nBest PR-AUC: **{pol['best_by_pr_auc']}** · cheapest at defaults: **{pol['cheapest_at_defaults']}** · "
                 f"best drift-triggered: **{pol['best_drift_policy']}**\n\n")
        for n, c in (("policy_timeline", "Weekly PR-AUC by policy"), ("policy_tradeoff", "Performance vs retraining effort, and cost"),
                     ("cost_sensitivity", "Cheapest policy under different cost assumptions")):
            s.append(fig(readme_dir, n, c))

    sh = load_json("shap_summary.json")
    if sh:
        s.append("### 7. Explainability with TreeSHAP (RQ4)\n")
        s.append(f"Importance rank correlation between the first and last 3 deployment weeks: "
                 f"**{sh['spearman_rank_corr_early_vs_late']:.2f}**; top-10 features in common: **{sh['top10_overlap']}/10**; "
                 f"top-5 features carry {sh['top5_share_pct_early']:.0f}% of total |SHAP| early and {sh['top5_share_pct_late']:.0f}% late.\n\n")
        s.append("Top-10 early: " + ", ".join(sh["top10_early"]) + "  \nTop-10 late: " + ", ".join(sh["top10_late"]) + "\n\n")
        for n, c in (("shap_global_top15", "Mean |SHAP|, early vs late"), ("shap_weekly_heatmap", "Share of |SHAP| over time"),
                     ("shap_beeswarm_late", "Feature effects in the last 3 weeks")):
            s.append(fig(readme_dir, n, c))

    body = "".join(s)
    nl = chr(10)
    out = []
    for line in body.split(nl):
        out.append(line)
        if line.startswith("### "):
            out.append("")
    body = nl.join(out)
    return body.replace("Do not report them.**" + nl, "Do not report them.**" + nl + nl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=str(config.ROOT / "README.md"))
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="permit writing synthetic-data results into the project README (never do this for real)")
    a = ap.parse_args()
    path = Path(a.readme)
    dq = load_json("data_quality.json")
    if dq and dq.get("data_source") == "synthetic" and path.resolve() == (config.ROOT / "README.md").resolve()             and not a.allow_synthetic:
        raise SystemExit("Refusing to write SYNTHETIC results into the project README. "
                         "Run on the real IEEE-CIS data, or pass --readme <other file>.")
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"markers {START} / {END} not found in {path}")
    body = build(path.parent)
    new = text[:text.index(START) + len(START)] + "\n\n" + body + "\n" + text[text.index(END):]
    path.write_text(new, encoding="utf-8")
    print("README results section updated ->", path)


if __name__ == "__main__":
    main()
