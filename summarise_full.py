"""
Summarise the four CSVs from analysis_full.py into paper-style tables
and figures.

Inputs:
  results_full.csv     — 22 models x 3 tickers x 3 horizons rel-MSE
  mcs_results.csv      — MCS membership at 75 % and 90 %
  ale_importance.csv   — ALE-based feature importance per (model, cell)
  var_results.csv      — VaR loss, Kupiec, Christoffersen per (model, ticker)

Outputs:
  tab_full_relmse.csv / .md      — paper Table 2/4/6 analogue
  tab_full_mcs.csv               — MCS membership table
  tab_full_var.csv / .md         — VaR summary
  fig_full_relmse.png            — bar chart, 22 models x 3 horizons
  fig_full_mcs.png               — MCS-90 membership grid
  fig_full_ale_importance.png    — ALE-based variable importance
  fig_full_var.png               — VaR diagnostics
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TICKERS = ["AAPL", "AMZN", "JPM"]
HORIZONS = ["h=1", "h=5", "h=22"]
HOR_LABEL = {"h=1": "1-day", "h=5": "1-week", "h=22": "1-month"}

# Paper-style 22-model order, matching paper Table 2 columns
MODEL_ORDER = [
    "HAR", "HAR-X", "LogHAR", "LevHAR", "SHAR", "HARQ",
    "Ridge", "Lasso", "ElasticNet", "AdaLasso", "PostLasso",
    "Bagging", "RandomForest", "GBoost",
    "NN1_1", "NN1_10", "NN2_1", "NN2_10",
    "NN3_1", "NN3_10", "NN4_1", "NN4_10",
]
SHORT = {
    "HAR": "HAR", "HAR-X": "HAR-X", "LogHAR": "LogHAR", "LevHAR": "LevHAR",
    "SHAR": "SHAR", "HARQ": "HARQ",
    "Ridge": "RR", "Lasso": "LA", "ElasticNet": "EN",
    "AdaLasso": "A-LA", "PostLasso": "P-LA",
    "Bagging": "BG", "RandomForest": "RF", "GBoost": "GB",
    "NN1_1": r"$\mathrm{NN}_1^1$", "NN1_10": r"$\mathrm{NN}_1^{10}$",
    "NN2_1": r"$\mathrm{NN}_2^1$", "NN2_10": r"$\mathrm{NN}_2^{10}$",
    "NN3_1": r"$\mathrm{NN}_3^1$", "NN3_10": r"$\mathrm{NN}_3^{10}$",
    "NN4_1": r"$\mathrm{NN}_4^1$", "NN4_10": r"$\mathrm{NN}_4^{10}$",
}
# Plain-text labels for CSV / markdown tables
SHORT_TXT = {
    **{k: v for k, v in {
        "HAR": "HAR", "HAR-X": "HAR-X", "LogHAR": "LogHAR", "LevHAR": "LevHAR",
        "SHAR": "SHAR", "HARQ": "HARQ",
        "Ridge": "RR", "Lasso": "LA", "ElasticNet": "EN",
        "AdaLasso": "A-LA", "PostLasso": "P-LA",
        "Bagging": "BG", "RandomForest": "RF", "GBoost": "GB",
    }.items()},
    "NN1_1": "NN1^1", "NN1_10": "NN1^10",
    "NN2_1": "NN2^1", "NN2_10": "NN2^10",
    "NN3_1": "NN3^1", "NN3_10": "NN3^10",
    "NN4_1": "NN4^1", "NN4_10": "NN4^10",
}

FAMILY = {
    **{m: "HAR family" for m in ["HAR", "HAR-X", "LogHAR", "LevHAR", "SHAR", "HARQ"]},
    **{m: "Regularised linear"
       for m in ["Ridge", "Lasso", "ElasticNet", "AdaLasso", "PostLasso"]},
    **{m: "Tree ensemble" for m in ["Bagging", "RandomForest", "GBoost"]},
    **{m: "Neural network"
       for m in ["NN1_1", "NN1_10", "NN2_1", "NN2_10",
                  "NN3_1", "NN3_10", "NN4_1", "NN4_10"]},
}
FAMILY_COLOR = {
    "HAR family":          "#34495e",
    "Regularised linear":  "#27ae60",
    "Tree ensemble":       "#c0392b",
    "Neural network":      "#e67e22",
}


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 240,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
})


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# 1. Paper-style relative-MSE table (paper Table 2/4/6 analogue)
# ---------------------------------------------------------------------------
def build_relmse_table(df):
    rows = []
    for h_lab in HORIZONS:
        for m in MODEL_ORDER:
            row = {"Horizon": HOR_LABEL[h_lab], "Model": SHORT_TXT[m]}
            cells = []
            for t in TICKERS:
                sub = df[(df.ticker == t) & (df.horizon == h_lab) &
                         (df.model == m)]
                if sub.empty:
                    row[t] = ""; cells.append(np.nan); continue
                r = sub.iloc[0]
                cell = f"{r['rel_mse']:.3f}{stars(r['p_value'])}"
                row[t] = cell
                cells.append(r["rel_mse"])
            row["Mean"] = f"{np.nanmean(cells):.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. MCS summary
# ---------------------------------------------------------------------------
def build_mcs_table(mcs):
    """Membership count: in how many of the 9 cells does each model appear in MCS_90?"""
    rows = []
    for m in MODEL_ORDER:
        sub = mcs[mcs.model == m]
        n90 = sub["in_MCS_90"].sum()
        n75 = sub["in_MCS_75"].sum()
        rows.append({"Model": SHORT_TXT[m],
                     "in_MCS_90 (/9)": int(n90),
                     "in_MCS_75 (/9)": int(n75),
                     "Family": FAMILY[m]})
    return pd.DataFrame(rows).sort_values(
        ["in_MCS_90 (/9)", "in_MCS_75 (/9)"], ascending=False)


# ---------------------------------------------------------------------------
# 3. VaR summary
# ---------------------------------------------------------------------------
def build_var_table(var_df):
    rows = []
    for m in MODEL_ORDER:
        sub = var_df[var_df.model == m]
        if sub.empty:
            continue
        rows.append({
            "Model": SHORT_TXT[m],
            "Hit rate (mean)": f"{sub.hit_rate.mean():.3f}",
            "Quantile loss x1e3 (mean)": f"{sub.quantile_loss.mean()*1e3:.3f}",
            "Kupiec p (mean)": f"{sub.p_kupiec.mean():.3f}",
            "Christoffersen p (mean)": f"{sub.p_christoffersen.mean():.3f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1: 22-model rel-MSE bar chart (3 horizons)
# ---------------------------------------------------------------------------
def fig_relmse(df, outpath):
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.2), sharex=True)
    x = np.arange(len(MODEL_ORDER))
    bar_colors = [FAMILY_COLOR[FAMILY[m]] for m in MODEL_ORDER]
    for i, h_lab in enumerate(HORIZONS):
        ax = axes[i]
        means = []
        for m in MODEL_ORDER:
            sub = df[(df.horizon == h_lab) & (df.model == m)]
            means.append(sub["rel_mse"].mean())
        ax.bar(x, means, color=bar_colors, alpha=0.88,
               edgecolor="#222222", linewidth=0.3)
        ax.axhline(1.0, color="black", lw=0.7, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("rel-MSE")
        ax.set_title(f"Horizon: {HOR_LABEL[h_lab]}")
        ax.set_ylim(0, max(2.6, max(means) * 1.05))
        # annotate the cross-section mean rel-MSE under each bar
        for xi, v in zip(x, means):
            ax.text(xi, v + 0.04, f"{v:.2f}",
                    ha="center", fontsize=6.8, color="#333333")
    # family-colour legend (proxy artists)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.88)
               for c in FAMILY_COLOR.values()]
    axes[0].legend(handles, list(FAMILY_COLOR.keys()),
                   loc="upper left", ncol=4, fontsize=8)
    fig.suptitle("Cross-section mean rel-MSE across 22 forecasting models "
                 "on M$_{HAR}$",
                 y=0.995, fontsize=11.5, weight="bold")
    plt.savefig(outpath); plt.close()
    return outpath


# ---------------------------------------------------------------------------
# Figure 2: MCS-90 membership grid
# ---------------------------------------------------------------------------
def fig_mcs(mcs, outpath):
    rows = []
    for m in MODEL_ORDER:
        for h_lab in HORIZONS:
            for t in TICKERS:
                sub = mcs[(mcs.ticker == t) & (mcs.horizon == h_lab) &
                          (mcs.model == m)]
                in90 = bool(sub.iloc[0]["in_MCS_90"]) if not sub.empty else False
                rows.append({"model": m, "ticker": t, "h_lab": h_lab,
                             "cell_label": f"{t}\n{HOR_LABEL[h_lab]}",
                             "in90": in90})
    g = pd.DataFrame(rows)
    cell_order = [f"{t}\n{HOR_LABEL[h]}" for t in TICKERS for h in HORIZONS]
    grid = (g.pivot(index="model", columns="cell_label", values="in90")
              .reindex(index=MODEL_ORDER, columns=cell_order))

    fig, ax = plt.subplots(figsize=(8.0, 7.5))
    ax.imshow(grid.values.astype(int), cmap="Greys",
              aspect="auto", vmin=0, vmax=1, alpha=0.85)
    ax.set_xticks(range(len(cell_order)))
    ax.set_xticklabels(cell_order, rotation=0, fontsize=8)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([SHORT[m] for m in MODEL_ORDER], fontsize=8)
    for i, m in enumerate(MODEL_ORDER):
        for j, c in enumerate(cell_order):
            v = grid.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, "■" if v else "·",
                        ha="center", va="center", fontsize=10,
                        color="white" if v else "#999999")
    # family banding on y-axis
    for i, m in enumerate(MODEL_ORDER):
        ax.text(-1.6, i, FAMILY[m][:3],
                ha="right", va="center", fontsize=6.5,
                color=FAMILY_COLOR[FAMILY[m]])
    ax.set_title("Model Confidence Set (90 %) — Hansen-Lunde-Nason (2011)\n"
                  "Filled cells: model survives elimination in that "
                  "(ticker, horizon) test", fontsize=10.5, pad=12)
    ax.set_xlabel("(Ticker, horizon)")
    ax.set_ylabel("Model")
    plt.savefig(outpath); plt.close()
    return outpath


# ---------------------------------------------------------------------------
# Figure 3: ALE-based importance (paper Figure 7 analogue, M_HAR features)
# ---------------------------------------------------------------------------
def fig_ale(ale, outpath):
    imp = (ale.groupby(["model", "feature"])["importance"].mean()
              .unstack().reindex(columns=["RVD", "RVW", "RVM"]))
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(imp.index))
    width = 0.25
    colors = {"RVD": "#1F4E79", "RVW": "#2E75B6", "RVM": "#9DC3E6"}
    for k, f in enumerate(["RVD", "RVW", "RVM"]):
        ax.bar(x + (k - 1) * width, imp[f].values, width,
               label=f, color=colors[f], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(imp.index, rotation=0)
    ax.set_ylabel("Variable-importance share (ALE-based)")
    ax.axhline(1 / 3, color="#888888", ls=":", lw=0.7,
                label="equal-weight baseline (1/3)")
    ax.legend(loc="upper right", ncol=4)
    ax.set_ylim(0, 1.0)
    ax.set_title("Accumulated Local Effects — variable importance share "
                  "on M$_{HAR}$ (3 stocks × 3 horizons, mean)",
                  fontsize=10.5)
    plt.savefig(outpath); plt.close()
    return outpath


# ---------------------------------------------------------------------------
# Figure 4: VaR diagnostics
# ---------------------------------------------------------------------------
def fig_var(var_df, outpath):
    summary = (var_df.groupby("model")
                    [["hit_rate", "p_kupiec", "p_christoffersen"]]
                    .mean().reindex(MODEL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    x = np.arange(len(summary.index))
    bar_colors = [FAMILY_COLOR[FAMILY[m]] for m in MODEL_ORDER]

    ax = axes[0]
    ax.bar(x, summary["hit_rate"].values, color=bar_colors,
           alpha=0.9, edgecolor="#222", linewidth=0.3)
    ax.axhline(0.05, color="red", lw=0.7, ls="--",
                label=r"Target $\alpha = 5\%$")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean VaR hit rate")
    ax.set_title("VaR(5 %) hit rate by model (mean across stocks)")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.bar(x - 0.2, summary["p_kupiec"].values, 0.4,
           color="#2E75B6", alpha=0.9, label="Kupiec p")
    ax.bar(x + 0.2, summary["p_christoffersen"].values, 0.4,
           color="#C00000", alpha=0.9, label="Christoffersen p")
    ax.axhline(0.10, color="black", lw=0.6, ls="--",
                label="10 % rejection")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean p-value")
    ax.set_title("Coverage tests — Kupiec (unconditional) & Christoffersen "
                  "(conditional)")
    ax.legend(loc="upper right", ncol=3)
    fig.suptitle("Value-at-Risk application (one-day-ahead, "
                  r"$\alpha = 5\%$)",
                  y=1.02, fontsize=11.5, weight="bold")
    plt.savefig(outpath); plt.close()
    return outpath


def build_combined_table(df):
    """Side-by-side M_HAR vs M_ALL comparison."""
    rows = []
    info_sets = sorted(df["info_set"].unique()) if "info_set" in df.columns else ["M_HAR"]
    for h_lab in HORIZONS:
        for m in MODEL_ORDER:
            row = {"Horizon": HOR_LABEL[h_lab], "Model": SHORT_TXT[m]}
            for iset in info_sets:
                cells = []
                for t in TICKERS:
                    sub = df[(df.ticker==t)&(df.horizon==h_lab)&
                             (df.model==m)&(df.get("info_set", pd.Series([iset]*len(df)))==iset)]
                    if sub.empty:
                        continue
                    cells.append(sub.iloc[0]["rel_mse"])
                if cells:
                    row[f"{iset} mean"] = f"{np.nanmean(cells):.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv("results_full.csv")
    mcs = pd.read_csv("mcs_results.csv")
    ale = pd.read_csv("ale_importance.csv")
    var_df = pd.read_csv("var_results.csv")

    info_sets_present = sorted(df["info_set"].unique()) if "info_set" in df.columns else ["M_HAR"]
    print(f"Detected info sets: {info_sets_present}")

    for iset in info_sets_present:
        df_iset = df[df.info_set == iset] if "info_set" in df.columns else df
        mcs_iset = mcs[mcs.info_set == iset] if "info_set" in mcs.columns else mcs
        var_iset = var_df[var_df.info_set == iset] if "info_set" in var_df.columns else var_df

        suffix = f"_{iset.lower()}" if iset != "M_HAR" else ""

        tab_rel = build_relmse_table(df_iset)
        tab_rel.to_csv(f"tab_full_relmse{suffix}.csv", index=False)
        with open(f"tab_full_relmse{suffix}.md", "w") as f:
            f.write(f"**Full-paper Table — {iset}.** Cross-section relative MSE for "
                    f"the 22-model contest on {iset} (paper analogue of Tables 2 / 4 / 6).\n\n")
            f.write(tab_rel.to_markdown(index=False))
        print(f"Saved: tab_full_relmse{suffix}.csv / .md")

        tab_mcs = build_mcs_table(mcs_iset)
        tab_mcs.to_csv(f"tab_full_mcs{suffix}.csv", index=False)
        print(f"Saved: tab_full_mcs{suffix}.csv")

        tab_var = build_var_table(var_iset)
        tab_var.to_csv(f"tab_full_var{suffix}.csv", index=False)
        with open(f"tab_full_var{suffix}.md", "w") as f:
            f.write(f"**Full-paper Table — {iset}.** VaR application diagnostics "
                    f"(paper analogue of Tables 8–9).\n\n")
            f.write(tab_var.to_markdown(index=False))
        print(f"Saved: tab_full_var{suffix}.csv / .md")

        fig_relmse(df_iset, f"fig_full_relmse{suffix}.png")
        print(f"Saved: fig_full_relmse{suffix}.png")
        fig_mcs(mcs_iset, f"fig_full_mcs{suffix}.png")
        print(f"Saved: fig_full_mcs{suffix}.png")
        fig_var(var_iset, f"fig_full_var{suffix}.png")
        print(f"Saved: fig_full_var{suffix}.png")

    # ALE: stays single-info-set (computed on M_HAR features by the pipeline)
    fig_ale(ale, "fig_full_ale_importance.png")
    print("Saved: fig_full_ale_importance.png")

    # M_HAR vs M_ALL combined comparison
    if len(info_sets_present) > 1:
        tab_comb = build_combined_table(df)
        tab_comb.to_csv("tab_full_mhar_vs_mall.csv", index=False)
        with open("tab_full_mhar_vs_mall.md", "w") as f:
            f.write("**M_HAR vs M_ALL comparison.** Cross-section mean rel-MSE per "
                    "horizon × model under each information set.\n\n")
            f.write(tab_comb.to_markdown(index=False))
        print("Saved: tab_full_mhar_vs_mall.csv / .md")


if __name__ == "__main__":
    main()
