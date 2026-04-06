import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# File paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "Final_Dataset_Dummy_ESG.csv")

# Intermediate CSV outputs
STEP0_CSV  = os.path.join(BASE_DIR, "Step0CleanPanelBase.csv")
STEP2_CSV  = os.path.join(BASE_DIR, "Step2PanelDataWithDummy.csv")

# Excel outputs (one workbook per regression step)
STEP1_XLSX = os.path.join(BASE_DIR, "Step1FEContinuousESG.xlsx")
STEP2_XLSX = os.path.join(BASE_DIR, "Step2FEDummyESG.xlsx")
STEP3_XLSX = os.path.join(BASE_DIR, "Step3PooledDummyResults.xlsx")
STEP4_XLSX = os.path.join(BASE_DIR, "Step4RobustnessLag.xlsx")
STEP5_XLSX = os.path.join(BASE_DIR, "Step5YearlyPooledDummy.xlsx")

# Consolidated final report (Excel + plain text)
FINAL_XLSX = os.path.join(BASE_DIR, "FinalAcademicReport.xlsx")
FINAL_TXT  = os.path.join(BASE_DIR, "FinalAcademicReport.txt")

# Utility functions — regression output formatting

def coef_table(res, vars_keep):
   
    out = pd.DataFrame({
        "coef":    res.params,
        "std_err": res.bse,
        "t":       res.tvalues,
        "p":       res.pvalues,
    })
    # Attach significance stars based on p-value thresholds
    out["sig"] = out["p"].apply(
        lambda x: "***" if x < 0.01 else ("**" if x < 0.05 else ("*" if x < 0.1 else ""))
    )
    # Restrict to the requested variables (in the specified order)
    out = out.loc[vars_keep].copy()
    # Combine coefficient and stars into a single display column
    out["coef_sig"] = out.apply(lambda r: f"{r['coef']:.4f}{r['sig']}", axis=1)
    out["std_err"]  = out["std_err"].map(lambda x: f"{x:.4f}")
    out["t"]        = out["t"].map(lambda x: f"{x:.2f}")
    out["p"]        = out["p"].map(lambda x: f"{x:.4f}")
    return out[["coef_sig", "std_err", "t", "p"]]


def write_regression_file(path, title, res, table_vars, notes, firms):

    with pd.ExcelWriter(path) as writer:
        # Meta sheet: summary of model specification
        meta = pd.DataFrame({
            "Item": ["Model", "Obs", "Firms", "TimeFE", "FirmFE", "SE"],
            "Value": [
                title,
                int(res.nobs),
                int(firms.nunique()),
                "Yes" if "C(Time)" in res.model.formula else "No",
                "Yes" if "C(Firm)" in res.model.formula else "No",
                "Cluster(Firm)" if getattr(res, "cov_type", "") == "cluster"
                else getattr(res, "cov_type", ""),
            ],
        })
        meta.to_excel(writer, sheet_name="Meta", index=False)
        # Regression sheet: formatted coefficient table
        coef_table(res, table_vars).to_excel(writer, sheet_name="Regression", index=True)
        # Notes sheet: specification description
        pd.DataFrame({"Notes": [notes]}).to_excel(writer, sheet_name="Notes", index=False)

# Utility functions — plain-text academic report formatting

def three_line_table(df, index=True, float_decimals=4):

    if df is None:
        return ""
    d = df.copy()

    def _fmt(x):
        """Format a single cell value as a string."""
        if pd.isna(x):
            return ""
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.{float_decimals}f}"
        return str(x)

    # Optionally prepend index as the first column
    if index:
        idx_name = d.index.name if d.index.name else ""
        d = d.reset_index().rename(columns={"index": idx_name if idx_name else "Variable"})
    else:
        d = d.reset_index(drop=True)

    # Apply formatting to all cells
    # (pandas 3.0 renamed applymap → map; handle both versions)
    if hasattr(d, 'map'):
        d = d.map(_fmt)
    elif hasattr(d, 'applymap'):
        d = d.applymap(_fmt)
    else:
        d = d.apply(lambda col: col.map(_fmt))

    cols = list(d.columns)
    rows = d.values.tolist()

    # Compute column widths as the maximum of header and cell lengths
    widths = [len(str(c)) for c in cols]
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(cell))

    def _join_row(items):
        return "  ".join(items[j].ljust(widths[j]) for j in range(len(items)))

    header     = _join_row([str(c) for c in cols])
    body_lines = [_join_row(r) for r in rows]
    total_width = len(header)

    # Three horizontal rules: top, below header, bottom
    rule = "-" * total_width
    return "\n".join([rule, header, rule, *body_lines, rule])


def model_fit_table(res):

    try:
        r2  = getattr(res, "rsquared",     np.nan)
        r2a = getattr(res, "rsquared_adj", np.nan)
        aic = getattr(res, "aic",          np.nan)
        bic = getattr(res, "bic",          np.nan)
        nobs = getattr(res, "nobs",        np.nan)
    except Exception:
        r2 = r2a = aic = bic = nobs = np.nan
    return pd.DataFrame({
        "Statistic": ["N", "R-squared", "Adj. R-squared", "AIC", "BIC"],
        "Value":     [nobs, r2, r2a, aic, bic],
    })


def safe_coef_table(res, vars_keep):

    available = [v for v in vars_keep if v in res.params.index]
    if not available:
        return pd.DataFrame()
    return coef_table(res, available)


def build_multi_model_table(title, res_list, var_lists, model_labels=None):

    n_models = len(res_list)
    if model_labels is None:
        model_labels = [f"Model({i + 1})" for i in range(n_models)]

    # Collect all unique variables in the order they first appear across models
    all_vars, seen = [], set()
    for vlist in var_lists:
        for v in vlist:
            if v not in seen:
                all_vars.append(v)
                seen.add(v)

    # Build coefficient/SE string pairs for each (variable, model) cell
    cells = {}
    for i, (res, vlist) in enumerate(zip(res_list, var_lists)):
        for var in vlist:
            try:
                coef = res.params[var]
                se   = res.bse[var]
                pval = res.pvalues[var]
                sig  = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
                cells[(var, i)] = (f"{coef:.4f}{sig}", f"({se:.4f})")
            except Exception:
                cells[(var, i)] = ("", "")

    # Determine column widths
    stat_labels = ["N", "R2", "Adj R2"]
    var_col_w = max(
        len("Variable"),
        max((len(v) for v in all_vars), default=8),
        max((len(s) for s in stat_labels), default=6),
    )
    model_col_w = []
    for i, label in enumerate(model_labels):
        w = len(label)
        for var in all_vars:
            c, s = cells.get((var, i), ("", ""))
            w = max(w, len(c), len(s))
        try:
            w = max(w, len(str(int(res_list[i].nobs))))
            w = max(w, len(f"{res_list[i].rsquared:.4f}"))
            w = max(w, len(f"{res_list[i].rsquared_adj:.4f}"))
        except Exception:
            pass
        model_col_w.append(w)

    # Total table width: variable column + separator + model columns
    total_w = var_col_w + sum(3 + w for w in model_col_w)

    def rule(char):
        return char * total_w

    def fmt_row(label, vals):
        """Format one data row: label left-justified, values right-justified."""
        parts = [label.ljust(var_col_w)]
        for i, v in enumerate(vals):
            parts.append(f" {v.rjust(model_col_w[i])} ")
        return "|".join(parts)

    # Assemble table lines
    lines = []
    lines.append(rule("="))
    lines.append(f"{title} \u2014 coef with stars, SE in parentheses")
    lines.append(rule("="))
    lines.append(rule("-"))
    lines.append(fmt_row("Variable", model_labels))
    lines.append(rule("-"))

    # Variable rows: coefficient on first line, SE in parentheses on second
    for var in all_vars:
        coef_vals = [cells.get((var, i), ("", ""))[0] for i in range(n_models)]
        se_vals   = [cells.get((var, i), ("", ""))[1] for i in range(n_models)]
        lines.append(fmt_row(var, coef_vals))
        lines.append(fmt_row("", se_vals))

    lines.append(rule("-"))

    # Footer: N, R², Adj R²
    n_vals, r2_vals, adjr2_vals = [], [], []
    for res in res_list:
        try:    n_vals.append(str(int(res.nobs)))
        except: n_vals.append("")
        try:    r2_vals.append(f"{res.rsquared:.4f}")
        except: r2_vals.append("")
        try:    adjr2_vals.append(f"{res.rsquared_adj:.4f}")
        except: adjr2_vals.append("")

    lines.append(fmt_row("N",      n_vals))
    lines.append(fmt_row("R2",     r2_vals))
    lines.append(fmt_row("Adj R2", adjr2_vals))
    lines.append(rule("="))

    return "\n".join(lines)

# Plain-text academic report writer

def write_final_txt(path, context):

    lines = []
    add   = lines.append

    # Section 1: Descriptive statistics
    add("Descriptive Statistics")
    add("=" * 72)
    add(three_line_table(context.get("desc_stats"), index=True))
    add("")

    # Section 2: Main regressions (Models 1, 2, 3)
    reg_blocks = context.get("reg_blocks", [])
    if reg_blocks:
        for i, b in enumerate(reg_blocks, start=1):
            var_list = list(b["coef_table"].index)
            add(build_multi_model_table(
                f"Main Regression - Model({i})",
                [b["res"]],
                [var_list],
                model_labels=["Estimate"],
            ))
            add("")

    # Section 3: Robustness check — lagged Model 3 
    lag_blocks = context.get("lag_blocks", [])
    if lag_blocks:
        # Use the third lag model (pooled OLS with lagged dummies); fall back
        # to the last available model if fewer than three are available
        idx = 2 if len(lag_blocks) >= 3 else len(lag_blocks) - 1
        b = lag_blocks[idx]
        var_list = list(b["coef_table"].index)
        add(build_multi_model_table(
            "Robustness (Lag1) - Model(3)",
            [b["res"]],
            [var_list],
            model_labels=["Estimate"],
        ))
        add("")

    # Section 4: Year-by-year pooled dummy analysis
    add("Year-by-Year Pooled Dummy Analysis")
    add("=" * 72)
    add("Summary")
    add(three_line_table(context.get("yearly_meta"), index=False))
    add("")
    add("Coefficient Comparison Across Years")
    add(three_line_table(context.get("yearly_comparison"), index=False))
    add("")

    yearly_results = context.get("yearly_results") or {}
    if yearly_results:
        add("Per-Year Results")
        add("=" * 72)
        default_vars = ["EDummyY", "SDummyY", "GDummyY", "Size", "Leverage"]
        for y in sorted(yearly_results.keys()):
            res      = yearly_results[y]
            coef_df  = safe_coef_table(res, default_vars)
            var_list = list(coef_df.index) if not coef_df.empty else []
            if var_list:
                add(build_multi_model_table(
                    f"Year {y}",
                    [res],
                    [var_list],
                    model_labels=["Estimate"],
                ))
            add("")

    # Write all lines to file
    txt = "\n".join([ln if ln is not None else "" for ln in lines])
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

# Main analysis routine

def main():
    """
    Execute the full regression analysis pipeline and write all outputs.
    """

    # Load and clean the analytical panel
    df = pd.read_csv(INPUT_FILE)

    # Ensure all numerical columns are parsed as float (not object)
    num_cols = [
        "ENVIRONMENTAL_PILLAR_SCORE", "SOCIAL_PILLAR_SCORE", "GOVERNANCE_PILLAR_SCORE",
        "ROA", "Total_Assets", "Total_Revenue", "Total_Liabilities", "Net_Income",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing identifier or any core numerical variable
    df = df.dropna(subset=["ISSUER_TICKER", "Year", "Half"] + num_cols).copy()

    # Remove economically implausible observations (assets and revenue must be positive)
    df = df[(df["Total_Assets"] > 0) & (df["Total_Revenue"] > 0)]

    # Construct regression variables 
    df["Size"]     = np.log(df["Total_Assets"])          # Natural log of total assets
    df["Leverage"] = df["Total_Liabilities"] / df["Total_Assets"]  # Total debt ratio
    df["Growth"]   = np.log(df["Total_Revenue"])         # Natural log of total revenue
                                                          # (descriptive only; excluded from
                                                          # regressions due to collinearity
                                                          # with Size; rho ≈ 0.904)

    # Replace infinite values produced by logarithms with NaN, then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["Size", "Leverage", "Growth"])

    # Panel identifiers
    df["Firm"]     = df["ISSUER_TICKER"].astype(str)  # Firm fixed-effect identifier
    df["Half_num"] = df["Half"].map({"H1": 1, "H2": 2}).astype(int)

    # Semiannual time index: ensures strict monotonic ordering within firms.
    # H1 2018 → 20181; H2 2018 → 20182; H1 2019 → 20191; etc.
    df["Time"] = (df["Year"].astype(int) * 10 + df["Half_num"]).astype(int)

    # Annual above-median ESG dummy variables
    # Each dummy equals 1 if the firm's pillar score exceeds the cross-sectional
    # median for that year, and 0 otherwise.  Using year-specific medians
    # controls for aggregate drift in MSCI's scoring methodology.
    for s, d in zip(
        ["ENVIRONMENTAL_PILLAR_SCORE", "SOCIAL_PILLAR_SCORE", "GOVERNANCE_PILLAR_SCORE"],
        ["EDummyY",                    "SDummyY",             "GDummyY"],
    ):
        med     = df.groupby("Year")[s].transform("median")
        df[d]   = (df[s] > med).astype(int)

    # Column lists for clean output
    base_cols = [
        "ISSUER_TICKER", "ISSUER_ISIN", "ISSUER_CNTRY_DOMICILE", "Firm",
        "Year", "Half", "Time",
        "ENVIRONMENTAL_PILLAR_SCORE", "SOCIAL_PILLAR_SCORE", "GOVERNANCE_PILLAR_SCORE",
        "ROA", "Total_Assets", "Total_Revenue", "Total_Liabilities", "Net_Income",
        "Size", "Leverage", "Growth",
    ]
    dummy_cols = ["EDummyY", "SDummyY", "GDummyY"]

    # Step 0: Save base panel 
    df_step0 = df[base_cols].copy()
    df_step0.to_csv(STEP0_CSV, index=False, encoding="utf-8-sig")

    # Step 1: Model 1 — TWFE with continuous ESG scores
    # Specification: ROA ~ E + S + G + Size + Leverage + C(Firm) + C(Time)
    # Standard errors clustered at the firm level to account for within-firm
    # serial correlation in residuals.
       res1 = smf.ols(
        "ROA ~ ENVIRONMENTAL_PILLAR_SCORE + SOCIAL_PILLAR_SCORE + GOVERNANCE_PILLAR_SCORE"
        " + Size + Leverage + C(Firm) + C(Time)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["Firm"]})

    write_regression_file(
        STEP1_XLSX, "FEContinuousESG", res1,
        ["ENVIRONMENTAL_PILLAR_SCORE", "SOCIAL_PILLAR_SCORE", "GOVERNANCE_PILLAR_SCORE",
         "Size", "Leverage"],
        "DV: ROA; FirmFE + TimeFE; SE clustered by firm",
        df["Firm"],
    )

    # Step 2: Save panel with dummies; estimate Model 2
    df_step2 = df[base_cols + dummy_cols].copy()
    df_step2.to_csv(STEP2_CSV, index=False, encoding="utf-8-sig")

    # Model 2 — TWFE with annual ESG dummy variables
    # Specification: ROA ~ EDummyY + SDummyY + GDummyY + Size + Leverage + C(Firm) + C(Time)
    # Same fixed-effects structure as Model 1; ESG measured as above/below-median
    # indicator to address cross-year score comparability concerns.
    res2 = smf.ols(
        "ROA ~ EDummyY + SDummyY + GDummyY + Size + Leverage + C(Firm) + C(Time)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["Firm"]})

    write_regression_file(
        STEP2_XLSX, "FEDummyESG", res2,
        ["EDummyY", "SDummyY", "GDummyY", "Size", "Leverage"],
        "DV: ROA; Dummy by year median; FirmFE + TimeFE; SE clustered by firm",
        df["Firm"],
    )

    # Step 3: Model 3 — Pooled OLS with ESG dummy variables 
    # Specification: ROA ~ EDummyY + SDummyY + GDummyY + Size + Leverage + C(Time)
    # Firm fixed effects are removed; cross-sectional variation is retained.
    # HC1 heteroskedasticity-robust standard errors (White, 1980).
    res3 = smf.ols(
        "ROA ~ EDummyY + SDummyY + GDummyY + Size + Leverage + C(Time)",
        data=df,
    ).fit(cov_type="HC1")

    write_regression_file(
        STEP3_XLSX, "PooledDummyESG", res3,
        ["EDummyY", "SDummyY", "GDummyY", "Size", "Leverage"],
        "DV: ROA; Pooled OLS; TimeFE; HC1 robust SE",
        df["Firm"],
    )

        # Step 4: Robustness — one-period lagged regressors
    # Each ESG variable and control is shifted one semiannual period within
    # firms.  This reduces mechanical simultaneity: ESG performance at t-1
    # predicts ROA at t.
        df_lag = df.sort_values(["Firm", "Time"]).copy()

    # Construct lagged columns for all ESG and control variables
    lag_vars = [
        "ENVIRONMENTAL_PILLAR_SCORE", "SOCIAL_PILLAR_SCORE", "GOVERNANCE_PILLAR_SCORE",
        "EDummyY", "SDummyY", "GDummyY",
        "Size", "Leverage",
    ]
    for c in lag_vars:
        df_lag[c + "Lag1"] = df_lag.groupby("Firm")[c].shift(1)

    # Drop rows missing any lagged ESG score or control
    df_lag = df_lag.dropna(subset=[
        "ENVIRONMENTAL_PILLAR_SCORELag1", "SOCIAL_PILLAR_SCORELag1",
        "GOVERNANCE_PILLAR_SCORELag1", "SizeLag1", "LeverageLag1",
    ])

    # Lagged Model 1a — TWFE with lagged continuous ESG scores
    res4a = smf.ols(
        "ROA ~ ENVIRONMENTAL_PILLAR_SCORELag1 + SOCIAL_PILLAR_SCORELag1"
        " + GOVERNANCE_PILLAR_SCORELag1 + SizeLag1 + LeverageLag1 + C(Firm) + C(Time)",
        data=df_lag,
    ).fit(cov_type="cluster", cov_kwds={"groups": df_lag["Firm"]})

    # Lagged Model 2b — TWFE with lagged ESG dummy variables
    res4b = smf.ols(
        "ROA ~ EDummyYLag1 + SDummyYLag1 + GDummyYLag1 + SizeLag1 + LeverageLag1"
        " + C(Firm) + C(Time)",
        data=df_lag,
    ).fit(cov_type="cluster", cov_kwds={"groups": df_lag["Firm"]})

    # Lagged Model 3c — Pooled OLS with lagged ESG dummy variables (main robustness check)
    res4c = smf.ols(
        "ROA ~ EDummyYLag1 + SDummyYLag1 + GDummyYLag1 + SizeLag1 + LeverageLag1 + C(Time)",
        data=df_lag,
    ).fit(cov_type="HC1")

    # Save all three lagged model variants to a single workbook
    with pd.ExcelWriter(STEP4_XLSX) as writer:
        pd.DataFrame({
            "Item":  ["Obs", "Firms", "Note"],
            "Value": [
                int(res4a.nobs),
                int(df_lag["Firm"].nunique()),
                "Lag1 within firm; sorted by Firm-Time",
            ],
        }).to_excel(writer, sheet_name="Meta", index=False)

        coef_table(
            res4a,
            ["ENVIRONMENTAL_PILLAR_SCORELag1", "SOCIAL_PILLAR_SCORELag1",
             "GOVERNANCE_PILLAR_SCORELag1", "SizeLag1", "LeverageLag1"],
        ).to_excel(writer, sheet_name="LagContinuousFE")

        coef_table(
            res4b,
            ["EDummyYLag1", "SDummyYLag1", "GDummyYLag1", "SizeLag1", "LeverageLag1"],
        ).to_excel(writer, sheet_name="LagDummyFE")

        coef_table(
            res4c,
            ["EDummyYLag1", "SDummyYLag1", "GDummyYLag1", "SizeLag1", "LeverageLag1"],
        ).to_excel(writer, sheet_name="LagDummyPooled")

    # Step 5: Year-by-year cross-sectional analysis (2018–2024)
    # For each calendar year, estimate a pooled OLS model on the subsample
    # of observations in that year, retaining half-year fixed effects.
    # Include overall F-tests and a joint F-test on the ESG dummies.
    # Only years with at least 30 observations are estimated.
    years_to_analyse = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    yearly_results   = {}  # {year: fitted result}
    yearly_meta      = []  # Summary statistics per year

    for year in years_to_analyse:
        df_year = df[df["Year"] == year].copy()

        if len(df_year) >= 30:  # Minimum sample size for reliable estimation
            try:
                # Cross-sectional pooled OLS with half-year fixed effects
                res_year = smf.ols(
                    "ROA ~ EDummyY + SDummyY + GDummyY + Size + Leverage + C(Half)",
                    data=df_year,
                ).fit(cov_type="HC1")
                yearly_results[year] = res_year

                # F-tests
                # (1) Overall model F-test: joint significance of all regressors
                overall_f = float(np.asarray(res_year.fvalue))
                overall_p = float(np.asarray(res_year.f_pvalue))

                # (2) Joint ESG F-test: H₀: EDummyY = SDummyY = GDummyY = 0
                #     Tests whether the ESG dummies are jointly significant
                esg_joint   = res_year.f_test("EDummyY = 0, SDummyY = 0, GDummyY = 0")
                esg_joint_f = float(np.asarray(esg_joint.fvalue))
                esg_joint_p = float(np.asarray(esg_joint.pvalue))

                # Store per-year summary metadata
                yearly_meta.append({
                    "Year":          year,
                    "Observations":  int(res_year.nobs),
                    "Firms":         int(df_year["Firm"].nunique()),
                    "R_squared":     f"{res_year.rsquared:.4f}",
                    "Adj_R_squared": f"{res_year.rsquared_adj:.4f}",
                    "Overall_F":     f"{overall_f:.2f}",
                    "Overall_F_p":   f"{overall_p:.4f}",
                    "ESG_Joint_F":   f"{esg_joint_f:.2f}",
                    "ESG_Joint_F_p": f"{esg_joint_p:.4f}",
                })
            except Exception as e:
                print(f"Year {year} regression failed: {e}")
        else:
            print(f"Year {year}: insufficient observations (n={len(df_year)}), skipping.")

    # Save year-by-year results to dedicated workbook
    with pd.ExcelWriter(STEP5_XLSX) as writer:
        # Summary metadata: one row per year
        pd.DataFrame(yearly_meta).to_excel(writer, sheet_name="YearlySummary", index=False)

        # Per-year coefficient tables
        for year, res in yearly_results.items():
            try:
                coef_table(
                    res, ["EDummyY", "SDummyY", "GDummyY", "Size", "Leverage"]
                ).to_excel(writer, sheet_name=f"Year{year}")
            except Exception:
                pass

        # Cross-year coefficient comparison table:
        # rows = variables, columns = years; cells contain coef + significance stars
        if yearly_results:
            comparison_data = []
            var_list = ["EDummyY", "SDummyY", "GDummyY", "Size", "Leverage"]
            for var in var_list:
                row = {"Variable": var}
                for year in years_to_analyse:
                    if year in yearly_results:
                        try:
                            coef = yearly_results[year].params[var]
                            pval = yearly_results[year].pvalues[var]
                            sig  = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
                            row[str(year)] = f"{coef:.4f}{sig}"
                        except Exception:
                            row[str(year)] = "N/A"
                    else:
                        row[str(year)] = "N/A"
                comparison_data.append(row)
            pd.DataFrame(comparison_data).to_excel(writer, sheet_name="YearlyComparison", index=False)

    # Step 6: Compile descriptive statistics and correlation matrix

    # Annual observation and firm counts (used to report panel composition)
    year_counts    = df.groupby("Year").agg(obs=("ROA", "size"), firms=("Firm", "nunique")).reset_index()
    year_counts_30 = year_counts[year_counts["obs"] >= 30].copy()  # Years with n ≥ 30

    # Descriptive statistics for the seven key variables
    desc_vars  = ["ROA", "ENVIRONMENTAL_PILLAR_SCORE", "SOCIAL_PILLAR_SCORE",
                  "GOVERNANCE_PILLAR_SCORE", "Size", "Leverage", "Growth"]
    desc_stats = df[desc_vars].describe().T

    # Correlation matrix with Pearson correlation and significance stars
    from scipy import stats as _stats
    corr_vars   = ["ROA", "ENVIRONMENTAL_PILLAR_SCORE", "SOCIAL_PILLAR_SCORE",
                   "GOVERNANCE_PILLAR_SCORE", "Size", "Leverage", "Growth"]
    corr_labels = ["ROA", "E", "S", "G", "Size", "Leverage", "Growth"]
    corr_rows   = []

    for i, (v1, l1) in enumerate(zip(corr_vars, corr_labels)):
        row = {"Variable": f"({i + 1}) {l1}"}
        for j, (v2, l2) in enumerate(zip(corr_vars, corr_labels)):
            if i == j:
                # Diagonal: perfect correlation with itself
                row[f"({j + 1})"] = "1"
            elif i > j:
                # Lower triangle: fill with Pearson r and significance stars
                r, p = _stats.pearsonr(df[v1], df[v2])
                sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                row[f"({j + 1})"] = f"{r:.3f}{sig}"
            else:
                # Upper triangle: leave blank (matrix is symmetric)
                row[f"({j + 1})"] = "-"
        corr_rows.append(row)

    corr_matrix_df = pd.DataFrame(corr_rows).set_index("Variable")
    corr_matrix_df.index.name = "Variable"

    # Step 7: Write the consolidated FinalAcademicReport.xlsx
    # One workbook containing all results for easy reference
    with pd.ExcelWriter(FINAL_XLSX) as writer:
        # Panel data sample (first 100 rows for inspection)
        df_step0.head(100).to_excel(writer, sheet_name="DataSample",        index=False)
        desc_stats.to_excel(         writer, sheet_name="DescriptiveStats")
        corr_matrix_df.to_excel(     writer, sheet_name="CorrelationMatrix")
        year_counts.to_excel(        writer, sheet_name="YearlyCountsAll",   index=False)
        year_counts_30.to_excel(     writer, sheet_name="YearlyCountsN30",   index=False)

        # Model 1 (TWFE, continuous ESG)
        coef_table(res1, ["ENVIRONMENTAL_PILLAR_SCORE", "SOCIAL_PILLAR_SCORE",
                          "GOVERNANCE_PILLAR_SCORE", "Size", "Leverage"]
                  ).to_excel(writer, sheet_name="BaselineFEContinuous")

        # Model 2 (TWFE, annual dummies)
        coef_table(res2, ["EDummyY", "SDummyY", "GDummyY", "Size", "Leverage"]
                  ).to_excel(writer, sheet_name="MainFEDummy")

        # Model 3 (Pooled OLS, annual dummies)
        coef_table(res3, ["EDummyY", "SDummyY", "GDummyY", "Size", "Leverage"]
                  ).to_excel(writer, sheet_name="MainPooledDummy")

        # Robustness: lagged continuous TWFE
        coef_table(res4a, ["ENVIRONMENTAL_PILLAR_SCORELag1", "SOCIAL_PILLAR_SCORELag1",
                           "GOVERNANCE_PILLAR_SCORELag1", "SizeLag1", "LeverageLag1"]
                  ).to_excel(writer, sheet_name="LagFEContinuous")

        # Robustness: lagged dummy TWFE
        coef_table(res4b, ["EDummyYLag1", "SDummyYLag1", "GDummyYLag1", "SizeLag1", "LeverageLag1"]
                  ).to_excel(writer, sheet_name="LagFEDummy")

        # Robustness: lagged pooled OLS (main robustness check reported in dissertation)
        coef_table(res4c, ["EDummyYLag1", "SDummyYLag1", "GDummyYLag1", "SizeLag1", "LeverageLag1"]
                  ).to_excel(writer, sheet_name="LagPooledDummy")

        # Year-by-year summary and coefficient comparison
        if yearly_meta:
            pd.DataFrame(yearly_meta).to_excel(writer, sheet_name="YearlyPooledSummary", index=False)
            if yearly_results:
                comparison_data_final = []
                for var in ["EDummyY", "SDummyY", "GDummyY", "Size", "Leverage"]:
                    row = {"Variable": var}
                    for year in years_to_analyse:
                        if year in yearly_results:
                            try:
                                coef = yearly_results[year].params[var]
                                pval = yearly_results[year].pvalues[var]
                                sig  = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
                                row[str(year)] = f"{coef:.4f}{sig}"
                            except Exception:
                                row[str(year)] = "N/A"
                        else:
                            row[str(year)] = "N/A"
                    comparison_data_final.append(row)
                pd.DataFrame(comparison_data_final).to_excel(
                    writer, sheet_name="YearlyCoefComparison", index=False
                )

    # Step 8: Write the plain-text academic report
    try:
        yearly_meta_df = pd.DataFrame(yearly_meta)
    except Exception:
        yearly_meta_df = pd.DataFrame()

    try:
        yearly_comparison_df = pd.DataFrame(comparison_data_final)
    except Exception:
        yearly_comparison_df = pd.DataFrame()

    context = {
        "desc_stats": desc_stats,
        "reg_blocks": [
            {
                "title": "Baseline FE (Continuous ESG)",
                "notes": "ROA ~ E + S + G + controls + FirmFE + TimeFE; SE clustered by firm.",
                "coef_table": coef_table(
                    res1, ["ENVIRONMENTAL_PILLAR_SCORE", "SOCIAL_PILLAR_SCORE",
                           "GOVERNANCE_PILLAR_SCORE", "Size", "Leverage"]
                ),
                "res": res1,
            },
            {
                "title": "Main FE (Yearly ESG Dummies)",
                "notes": "ROA ~ EDummyY + SDummyY + GDummyY + controls + FirmFE + TimeFE; SE clustered by firm.",
                "coef_table": coef_table(res2, ["EDummyY", "SDummyY", "GDummyY", "Size", "Leverage"]),
                "res": res2,
            },
            {
                "title": "Pooled OLS (Yearly ESG Dummies)",
                "notes": "ROA ~ EDummyY + SDummyY + GDummyY + controls + TimeFE; HC1 robust SE.",
                "coef_table": coef_table(res3, ["EDummyY", "SDummyY", "GDummyY", "Size", "Leverage"]),
                "res": res3,
            },
        ],
        "lag_blocks": [
            {
                "title": "Lag FE (Continuous ESG, Lag1)",
                "notes": "Lag1 within firm; sorted by Firm-Time; SE clustered by firm.",
                "coef_table": coef_table(
                    res4a, ["ENVIRONMENTAL_PILLAR_SCORELag1", "SOCIAL_PILLAR_SCORELag1",
                            "GOVERNANCE_PILLAR_SCORELag1", "SizeLag1", "LeverageLag1"]
                ),
                "res": res4a,
            },
            {
                "title": "Lag FE (Yearly ESG Dummies, Lag1)",
                "notes": "Lag1 within firm; SE clustered by firm.",
                "coef_table": coef_table(
                    res4b, ["EDummyYLag1", "SDummyYLag1", "GDummyYLag1", "SizeLag1", "LeverageLag1"]
                ),
                "res": res4b,
            },
            {
                "title": "Lag Pooled OLS (Yearly ESG Dummies, Lag1)",
                "notes": "Lag1 within firm; TimeFE; HC1 robust SE.",
                "coef_table": coef_table(
                    res4c, ["EDummyYLag1", "SDummyYLag1", "GDummyYLag1", "SizeLag1", "LeverageLag1"]
                ),
                "res": res4c,
            },
        ],
        "yearly_meta":       yearly_meta_df,
        "yearly_comparison": yearly_comparison_df,
        "yearly_results":    yearly_results,
    }

    write_final_txt(FINAL_TXT, context)
    print(f"Plain-text academic report written: {FINAL_TXT}")
    print("Analysis complete.")
    print(f"Year-by-year results saved to: {STEP5_XLSX}")
    print(f"Consolidated report saved to:  {FINAL_XLSX}")


if __name__ == "__main__":
    main()
