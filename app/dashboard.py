import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("FRAUD_RESULTS_DIR", ROOT / "results"))
FIG, TAB = RES / "figures", RES / "tables"
BLUE, RED, GREEN, ORANGE, GRAY, PURPLE = "#2b6cb0", "#c53030", "#2f855a", "#dd6b20", "#718096", "#6b46c1"

st.set_page_config(page_title="Fraud Drift Lab", page_icon="🛡️", layout="wide")


@st.cache_data
def jload(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def csv(rel, **kw):
    p = RES / rel
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data
def stream(policy):
    p = RES / f"stream_{policy}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def policies():
    return sorted(p.stem.replace("window_metrics_", "") for p in TAB.glob("window_metrics_*.csv")) if TAB.exists() else []


def show_fig(name, caption=None):
    p = FIG / f"{name}.png"
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.caption(f"_{name}.png not generated yet_")


def kpi(col, label, value, help=None, delta=None):
    col.metric(label, value, delta=delta, help=help)


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.12),
                      template="plotly_white")
    return fig


dq, eda = jload("data_quality.json"), jload("eda_stats.json")
base, drift = jload("baseline_summary.json"), jload("drift_summary.json")
pol_sum, shap_sum = jload("policy_summary.json"), jload("shap_summary.json")

if dq is None:
    st.error("No results found. Run `python run_midsem.py` first.")
    st.stop()
if dq.get("data_source") == "synthetic":
    st.warning("Showing SYNTHETIC smoke-test results - not IEEE-CIS. Do not present these numbers.")

PAGES = ["🏠 Overview", "📊 Data & EDA", "▶️ Live replay", "🌊 Drift", "🔁 Policies & cost", "🔍 Explainability", "🧭 Method"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("MLOps-Integrated Fraud Detection with Drift-Aware Retraining\n\nMD3135 Data Science · Team lead: Aditya Katkar")

if page == PAGES[0]:
    st.title("🛡️ Fraud detection under concept drift")
    st.markdown("**Question:** a fraud model is trained once and deployed. Fraud tactics change. *How fast does it decay, "
                "can we detect it, and when is retraining worth its cost?*  We replay the IEEE-CIS transactions in "
                "chronological weekly windows to find out.")
    c = st.columns(5)
    kpi(c[0], "Transactions", f"{dq['final_shape'][0]:,}")
    kpi(c[1], "Fraud rate", f"{dq['fraud_rate_pct']}%", "Severe class imbalance")
    kpi(c[2], "Time span", f"{dq['time_span_days']:.0f} days")
    kpi(c[3], "Weekly windows", base["data"]["n_windows"] if base else "-")
    kpi(c[4], "Features used", base["data"]["n_selected_features"] if base else "-",
        f"from {base['data']['n_candidate_features']} candidates" if base else None)

    st.subheader("Headline findings")
    if base:
        prot = pd.DataFrame(base["evaluation_protocols"]).set_index("protocol")
        static = prot.loc["Random 80/20 split (static)", "pr_auc"]
        dep = prot.loc["Deploy-once, no retraining (pooled)", "pr_auc"]
        nr = base["policies"]["no_retrain"]["decay"]["pr_auc"]
        c = st.columns(3)
        kpi(c[0], "RQ1 · Static split overstates", f"{100 * (static - dep) / static:.0f}%",
            f"PR-AUC {static:.2f} on a random split vs {dep:.2f} when deployed once")
        kpi(c[1], "RQ1 · PR-AUC drop, never retrained", f"{nr['relative_drop_pct']:.1f}%",
            f"first 3 vs last 3 weeks · Spearman ρ={nr['spearman_rho']:.2f}, p={nr['spearman_p']:.3f}")
        if drift:
            kpi(c[2], "Covariate drift · adversarial AUC",
                f"{drift['adversarial_auc']['first_deployment_week']:.2f} → {drift['adversarial_auc']['last_deployment_week']:.2f}",
                "0.5 = later weeks look identical to the training period; 1.0 = perfectly distinguishable")
    if pol_sum:
        t = pd.DataFrame(pol_sum["policies"])
        t = t[t.label_delay_days == 0]
        nr_row = t[t.policy == "no_retrain"].iloc[0]
        best = t.sort_values("mean_pr_auc", ascending=False).iloc[0]
        cheap = t.sort_values("total_cost").iloc[0]
        c = st.columns(3)
        kpi(c[0], "RQ3 · Best PR-AUC policy", best.policy, f"+{best.pr_auc_gain_vs_no_retrain_pct:.1f}% vs never retraining "
            f"with {int(best.n_retrains)} retrains")
        kpi(c[1], "RQ3 · Cheapest at default costs", cheap.policy,
            f"{cheap.cost_saving_vs_no_retrain_pct:.1f}% cheaper than never retraining")
        kpi(c[2], "Drift-triggered best", pol_sum["best_drift_policy"] or "-")
    st.subheader("Research questions")
    st.table(pd.DataFrame({
        "RQ": ["RQ1", "RQ2", "RQ3", "RQ4"],
        "Question": ["How much does a never-retrained model degrade, and how optimistic is a static split?",
                     "Can ADWIN / DDM detect the degradation — how fast, how many false alarms?",
                     "Which policy (none / fixed / drift-triggered) gives the best performance-vs-cost trade-off?",
                     "Which features drive predictions, and do they change with drift?"],
        "Page": ["Live replay, Data", "Drift", "Policies & cost", "Explainability"]}))
    show_fig("decay_curve", "The never-retrained model over deployment weeks")

elif page == PAGES[1]:
    st.title("📊 Data & exploratory analysis")
    c = st.columns(4)
    kpi(c[0], "Rows × columns", f"{dq['final_shape'][0]:,} × {dq['final_shape'][1]}")
    kpi(c[1], "Fraud transactions", f"{dq['n_fraud']:,}")
    kpi(c[2], "Missing values", f"{dq['overall_missing_pct']}%")
    kpi(c[3], "With identity record", f"{dq['identity_coverage_pct']}%")
    if eda:
        c = st.columns(4)
        kpi(c[0], "Imbalance", f"{eda['imbalance_ratio']}:1")
        kpi(c[1], "Weekly fraud rate range", f"{eda['weekly_fraud_rate_pct_min']}–{eda['weekly_fraud_rate_pct_max']}%")
        if "median_amount_fraud" in eda:
            kpi(c[2], "Median amount · fraud", f"{eda['median_amount_fraud']}", f"legit: {eda['median_amount_legit']}")
        if "fraud_rate_with_identity_pct" in eda:
            kpi(c[3], "Fraud rate with / without identity",
                f"{eda['fraud_rate_with_identity_pct']}% / {eda['fraud_rate_without_identity_pct']}%")
    names = {"Fraud rate over time": "eda_fraud_over_time", "Fraud pattern shift": "eda_pattern_shift",
             "Fraud by category": "eda_fraud_by_category", "Amounts": "eda_amount", "Amount over time": "eda_amount_over_time",
             "Class balance": "eda_class_balance", "Missingness": "eda_missingness", "Daily cycle": "eda_hourly",
             "Correlation with target": "eda_target_correlation"}
    tabs = st.tabs(list(names))
    for t, n in zip(tabs, names.values()):
        with t:
            show_fig(n)
    ab = csv("tables/imbalance_ablation.csv")
    if ab is not None:
        st.subheader("Class-imbalance handling")
        st.dataframe(ab.round(4), hide_index=True, width="stretch")
        st.caption("Trained on the initial window, scored on every later week. On this data plain training was not worse than "
                   "class weighting or SMOTE — worth stating honestly in the paper.")

elif page == PAGES[2]:
    st.title("▶️ Live replay of deployment")
    st.caption("Press play to watch the model face one new week of transactions at a time — exactly as it would in production.")
    pols = policies()
    if not pols:
        st.stop()
    c0, c1 = st.columns([2, 1])
    default = pols.index("no_retrain") if "no_retrain" in pols else 0
    pol = c0.selectbox("Retraining policy", pols, index=default)
    metric = c1.selectbox("Timeline metric", ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "f1"])
    wm = csv(f"tables/window_metrics_{pol}.csv")
    st_ = stream(pol)
    runinfo = (jload(f"runs/{pol}.json") or {}) if (RES / f"runs/{pol}.json").exists() else {}
    retr = runinfo.get("retrain_windows") or ((base or {}).get("policies", {}).get(pol, {}).get("run", {}).get("retrain_windows", []))
    wmin, wmax = int(wm.window.min()), int(wm.window.max())
    week = st.slider("Deployment week", wmin, wmax, wmin, key="week")
    play = st.button("▶ Play from this week")
    holder = st.empty()
    ref_val = wm[metric].head(3).mean()
    thr = st.sidebar.slider("Alert threshold (live re-scoring)", 0.0, 1.0, float(wm["threshold"].iloc[0]), 0.01,
                            help="Flag a transaction when its fraud score exceeds this. Default = the model's own F1-optimal threshold.")

    def render(w, tick):
        r = wm[wm.window == w].iloc[0]
        s = st_[st_.window == w]
        flagged = s.proba >= thr
        tp = int(((s.y == 1) & flagged).sum()); fp = int(((s.y == 0) & flagged).sum()); fn = int(((s.y == 1) & ~flagged).sum())
        amt_missed = float(s.amount[(s.y == 1) & ~flagged].sum())
        with holder.container():
            c = st.columns(6)
            kpi(c[0], "Week", f"{w}", f"days {w * 7}–{w * 7 + 6}")
            kpi(c[1], "Transactions", f"{len(s):,}")
            kpi(c[2], "Frauds", f"{int(s.y.sum())}")
            kpi(c[3], f"{metric}", f"{r[metric]:.3f}", delta=f"{r[metric] - ref_val:+.3f} vs first weeks")
            kpi(c[4], "Caught / missed", f"{tp} / {fn}", f"${amt_missed:,.0f} of fraud missed")
            kpi(c[5], "False alarms", f"{fp}", f"precision {tp / max(tp + fp, 1):.2f}")
            upto = wm[wm.window <= w]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wm.window, y=wm[metric], mode="lines", line=dict(color="#cbd5e0", dash="dot"), name="full run"))
            fig.add_trace(go.Scatter(x=upto.window, y=upto[metric], mode="lines+markers", line=dict(color=BLUE), name=metric))
            fig.add_hline(y=ref_val, line=dict(color=GRAY, dash="dash"), annotation_text="early-weeks level")
            for rw in retr:
                if rw <= w:
                    fig.add_vline(x=rw, line=dict(color=GREEN, dash="dot"))
                    fig.add_annotation(x=rw, y=1, yref="paper", text="retrained", showarrow=False, font=dict(color=GREEN, size=10))
            fig.add_vline(x=w, line=dict(color=RED))
            fig.update_layout(title=f"{pol}: {metric} by week", xaxis_title="Deployment week")
            st.plotly_chart(style(fig, 330), width="stretch", key=f"tl_{tick}")
            a, b = st.columns(2)
            h = go.Figure()
            h.add_trace(go.Histogram(x=s.proba[s.y == 0], nbinsx=50, name="legitimate", marker_color=BLUE, opacity=0.6))
            h.add_trace(go.Histogram(x=s.proba[s.y == 1], nbinsx=50, name="fraud", marker_color=RED, opacity=0.6))
            h.add_vline(x=thr, line=dict(color="black", dash="dash"))
            h.update_layout(barmode="overlay", title="Fraud scores this week", yaxis_type="log", xaxis_title="model score")
            a.plotly_chart(style(h, 320), width="stretch", key=f"h_{tick}")
            feed = s.sort_values("proba", ascending=False).head(15).copy()
            feed["verdict"] = np.where(feed.proba >= thr, np.where(feed.y == 1, "✅ caught fraud", "⚠️ false alarm"),
                                       np.where(feed.y == 1, "❌ missed fraud", "ok"))
            feed["score"] = feed.proba.round(3); feed["amount"] = feed.amount.round(2)
            b.markdown("**Highest-risk transactions this week**")
            b.dataframe(feed[["score", "amount", "verdict"]], hide_index=True, width="stretch", height=320)

    if play:
        for i, w in enumerate(range(week, wmax + 1)):
            render(w, f"p{i}")
            time.sleep(0.7)
    else:
        render(week, "static")

elif page == PAGES[3]:
    st.title("🌊 Drift: is the world changing under the model?")
    if drift:
        c = st.columns(4)
        kpi(c[0], "Degradation onset (reference)", f"week {drift['degradation_onset_window']}" if drift["degradation_onset_window"] is not None else "none")
        kpi(c[1], "Adversarial AUC", f"{drift['adversarial_auc']['first_deployment_week']:.2f} → {drift['adversarial_auc']['last_deployment_week']:.2f}")
        kpi(c[2], "Features with PSI>0.25 (last wk)", drift["psi"]["n_features_psi_over_0_25_in_last_window"])
        kpi(c[3], "Most shifted", next(iter(drift["psi"]["most_shifted_features"])))
    t1, t2, t3 = st.tabs(["Detector alarms", "Feature shift (PSI)", "Adversarial validation"])
    with t1:
        det = csv("tables/drift_detections.csv")
        wm = csv("tables/window_metrics_no_retrain.csv")
        if det is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wm.window, y=wm.pr_auc, name="PR-AUC (never retrained)", line=dict(color=RED), yaxis="y2"))
            for i, (name, g) in enumerate(det.groupby("detector")):
                cnt = g.groupby("window").size()
                fig.add_trace(go.Scatter(x=cnt.index, y=[name] * len(cnt), mode="markers", name=name,
                                         marker=dict(size=8 + 4 * np.log1p(cnt.values), opacity=0.7)))
            if drift and drift["degradation_onset_window"] is not None:
                fig.add_vline(x=drift["degradation_onset_window"], line=dict(color="black"), annotation_text="onset")
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="PR-AUC"), xaxis_title="Deployment week")
            st.plotly_chart(style(fig, 420), width="stretch")
            st.info("**Honest reading:** ADWIN/DDM run on the raw error stream of a model that is never retrained fire early and often "
                    "(the stream is very imbalanced and noisy), often before degradation is measurable. That is why the drift-triggered "
                    "policy adds increase-only filtering, a cool-down and a detector reset after each retrain (see *Policies & cost*).")
            st.dataframe(pd.DataFrame(drift["detectors"]).T[["n_detections", "first_detection_window", "delay_vs_onset_weeks"]],
                         width="stretch")
    with t2:
        psi = csv("tables/psi_top_features.csv", index_col=0)
        if psi is not None:
            fig = px.imshow(psi.T.clip(0, 1), aspect="auto", color_continuous_scale="magma_r", labels=dict(x="Week", y="", color="PSI"))
            st.plotly_chart(style(fig, 520), width="stretch")
            f = st.selectbox("Zoom on one feature", list(psi.columns))
            line = px.line(psi[f], labels=dict(index="Week", value="PSI"))
            line.add_hline(y=0.25, line_dash="dash", line_color=RED, annotation_text="major shift (0.25)")
            line.add_hline(y=0.10, line_dash="dot", line_color=ORANGE, annotation_text="moderate (0.10)")
            st.plotly_chart(style(line, 300), width="stretch")
    with t3:
        adv = csv("tables/adversarial_drift.csv")
        if adv is not None:
            fig = px.line(adv, x="window", y="adversarial_auc", markers=True)
            fig.add_hline(y=0.5, line_dash="dash", line_color=GRAY, annotation_text="identical to training period")
            st.plotly_chart(style(fig, 360), width="stretch")
            st.caption("A classifier is trained to tell the reference period from week *w*. AUC→1 means the incoming data no longer "
                       "looks like the training data (covariate drift).")

elif page == PAGES[4]:
    st.title("🔁 Retraining policies and their cost")
    pc = csv("policy_comparison.csv")
    if pc is None:
        st.info("Run `python -m src.policy_comparison` to generate this page.")
        st.stop()
    main = pc[pc.label_delay_days == 0].copy()
    st.markdown("**Set your own cost assumptions** — the ranking of policies is recomputed instantly.")
    c = st.columns(3)
    review = c[0].slider("Review cost per flagged transaction ($)", 0.0, 20.0, 2.0, 0.5)
    retrain = c[1].slider("Cost of one retraining ($)", 0.0, 20_000.0, 500.0, 50.0)
    loss = c[2].slider("Share of a missed fraud amount that is lost", 0.0, 1.0, 1.0, 0.05)
    for d in (main, pc):
        d["c_loss"] = loss * d.missed_fraud_amount
        d["c_review"] = review * d.flagged
        d["c_retrain"] = retrain * d.n_retrains
        d["c_total"] = d.c_loss + d.c_review + d.c_retrain
    base_cost = main.loc[main.policy == "no_retrain", "c_total"].iloc[0]
    main["saving_pct"] = 100 * (base_cost - main.c_total) / base_cost
    best = main.sort_values("c_total").iloc[0]
    c = st.columns(3)
    kpi(c[0], "Cheapest policy", best.policy, f"{best.saving_pct:.1f}% cheaper than never retraining")
    kpi(c[1], "Best PR-AUC", main.sort_values("mean_pr_auc").iloc[-1].policy)
    kpi(c[2], "Least retraining for gain", main[main.n_retrains > 0].sort_values("mean_pr_auc").iloc[-1].policy
        if (main.n_retrains > 0).any() else "-")
    m = main.sort_values("c_total")
    fig = go.Figure()
    for col, lab, colr in (("c_loss", "Missed fraud", RED), ("c_review", "Review", ORANGE), ("c_retrain", "Retraining", BLUE)):
        fig.add_trace(go.Bar(y=m.policy, x=m[col], name=lab, orientation="h", marker_color=colr))
    fig.update_layout(barmode="stack", yaxis=dict(autorange="reversed"), title="Total cost by policy", xaxis_title="$")
    a, b = st.columns(2)
    a.plotly_chart(style(fig, 420), width="stretch")
    fig2 = px.scatter(main, x="n_retrains", y="mean_pr_auc", text="policy", size=np.full(len(main), 14),
                      color=main.policy.str.startswith("drift_").map({True: "drift-triggered", False: "baseline / fixed"}),
                      color_discrete_map={"drift-triggered": GREEN, "baseline / fixed": BLUE})
    fig2.update_traces(textposition="top center")
    fig2.update_layout(title="Performance vs retraining effort", xaxis_title="# retrainings", yaxis_title="mean weekly PR-AUC")
    b.plotly_chart(style(fig2, 420), width="stretch")
    cols = ["policy", "n_retrains", "mean_pr_auc", "pr_auc_gain_vs_no_retrain_pct", "wilcoxon_p_vs_no_retrain",
            "mean_recall_at_1pct_fpr", "train_seconds", "c_total", "saving_pct", "break_even_retrain_cost"]
    st.dataframe(main[cols].round(3), hide_index=True, width="stretch")
    st.caption("**Break-even retrain cost** = the price per retraining above which the policy costs more than never retraining. "
               "It does not depend on the cost assumptions you chose, only on review cost and loss share. "
               "Wilcoxon p compares per-week PR-AUC against never retraining (paired by week).")
    show_fig("policy_timeline"); show_fig("cost_sensitivity")
    vs = csv("policy_stats_vs_no_retrain.csv")
    if vs is not None:
        st.subheader("How sure are we? Gains with confidence intervals")
        a, b = st.columns([3, 2])
        with a:
            show_fig("policy_forest")
        with b:
            mb = csv("policy_stats_matched_budget.csv")
            st.markdown("**Drift-triggered vs fixed interval, same number of retrainings**")
            if mb is not None and len(mb):
                st.dataframe(mb[["triggered", "fixed", "n_retrains", "mean_diff", "ci_low", "ci_high"]].round(3),
                             hide_index=True, width="stretch")
                st.caption("Every interval contains 0: at equal retraining effort we cannot tell drift-triggered from fixed-interval retraining.")
            st.caption(f"{int(vs.significant_holm_5pct.sum())} of {len(vs)} policies beat never retraining after Holm correction.")
    ex = csv("extra_experiments.csv")
    if ex is not None:
        st.subheader("Do the conclusions survive other design choices?")
        st.dataframe(ex[["group", "variant", "policy", "mean_pr_auc", "n_retrains", "pr_auc_gain_pct", "cost_saving_pct"]].round(3),
                     hide_index=True, width="stretch")
        show_fig("extra_experiments")
    d = pc[pc.policy.str.contains("delay") | (pc.policy == "fixed_4w")]
    if len(d) > 1:
        st.subheader("Label-delay sensitivity (fixed 4-week retraining)")
        st.dataframe(d[["policy", "label_delay_days", "mean_pr_auc", "mean_recall_at_1pct_fpr", "c_total"]].round(3),
                     hide_index=True, width="stretch")
        st.caption("Real fraud labels arrive days or weeks late (chargebacks). Longer delay means the retrained model is trained on older data.")

elif page == PAGES[5]:
    st.title("🔍 Explainability (TreeSHAP)")
    tab = csv("tables/shap_early_vs_late.csv", index_col=0)
    byweek = csv("tables/shap_mean_abs_by_week.csv", index_col=0)
    if tab is None:
        st.info("Run `python -m src.explain` to generate this page.")
        st.stop()
    if shap_sum:
        c = st.columns(3)
        kpi(c[0], "Importance rank correlation early vs late", f"{shap_sum['spearman_rank_corr_early_vs_late']:.2f}")
        kpi(c[1], "Top-10 features in common", f"{shap_sum['top10_overlap']} / 10")
        kpi(c[2], "Top-5 share of total |SHAP|", f"{shap_sum['top5_share_pct_early']:.0f}% → {shap_sum['top5_share_pct_late']:.0f}%")
    top = tab.head(15).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(y=top.feature, x=top.early_mean_abs_shap, name="first 3 weeks", orientation="h", marker_color=BLUE))
    fig.add_trace(go.Bar(y=top.feature, x=top.late_mean_abs_shap, name="last 3 weeks", orientation="h", marker_color=RED))
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"), title="Mean |SHAP| of the deployed model")
    st.plotly_chart(style(fig, 560), width="stretch")
    f = st.selectbox("Follow one feature through time", list(tab.index[:40]))
    st.plotly_chart(style(px.line(byweek[f], labels=dict(index="Week", value="mean |SHAP|"), markers=True), 300), width="stretch")
    a, b = st.columns(2)
    with a:
        show_fig("shap_weekly_heatmap")
    with b:
        show_fig("shap_beeswarm_late")
    st.caption("Features are anonymised by the dataset owner (V*, C*, D*, id_*), so explanations describe *which* signals matter and "
               "*when*, not their business meaning.")

else:
    st.title("🧭 Method")
    st.markdown("""
**Simulation.** Transactions are sorted by time and cut into 7-day windows. The model is trained on the first 8 windows, then each
later window is scored using only data available before it. A retraining policy decides whether to retrain before the next window.

**Leakage control.** `TransactionDT` is never a feature · feature selection is fitted on the training window only · the decision
threshold is tuned on the last training window, never on deployment data · a retrain at week *w* sees only weeks < *w* (optionally delayed).

**Policies.** *No retraining* · *Fixed interval* (every k weeks) · *Drift-triggered* (ADWIN or DDM on the error or fraud-miss
stream, or a weekly PR-AUC-drop rule; increase-only, cool-down, detector reset after each retrain).

**Cost.** `fraud_loss·missed_amount + review·flagged + retrain·n_retrains`, plus an assumption-free break-even retraining cost.

**Metrics.** PR-AUC (primary), ROC-AUC, recall at 1% false-positive rate, precision/recall/F1 at a fixed threshold.
""")
    if base:
        st.json(base["data"])
