# -*- coding: utf-8 -*-
"""Build a table of used firms: ISIN, company name, and country.

Outputs:
- UsedCompany_ISIN_Name_Country.csv
- UsedCompany_ISIN_Name_Country.xlsx
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_CSV = os.path.join(BASE_DIR, "Step2PanelDataWithDummy.csv")
GLOBAL_XLSX = os.path.join(BASE_DIR, "Global ISIN.xlsx")
US_XLSX = os.path.join(BASE_DIR, "U.S. Ticker.xlsx")
MSCI_CSV = os.path.join(BASE_DIR, "MSCI.csv")

OUT_CSV = os.path.join(BASE_DIR, "UsedCompany_ISIN_Name_Country.csv")
OUT_XLSX = os.path.join(BASE_DIR, "UsedCompany_ISIN_Name_Country.xlsx")


def load_panel_companies() -> pd.DataFrame:
    cols = ["ISSUER_TICKER", "ISSUER_ISIN", "ISSUER_CNTRY_DOMICILE"]
    df = pd.read_csv(PANEL_CSV, usecols=cols)
    df = df.rename(
        columns={
            "ISSUER_TICKER": "Ticker",
            "ISSUER_ISIN": "ISIN",
            "ISSUER_CNTRY_DOMICILE": "Country",
        }
    )
    df = df.dropna(subset=["ISIN", "Country"]).drop_duplicates()
    return df


def map_name_from_global(panel_df: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_excel(GLOBAL_XLSX, usecols=["isin", "conm"])
    g = g.rename(columns={"isin": "ISIN", "conm": "CompanyName"})
    g["ISIN"] = g["ISIN"].astype(str).str.strip().str.upper()
    g["CompanyName"] = g["CompanyName"].astype(str).str.strip()
    g = g.dropna(subset=["ISIN"]).drop_duplicates(subset=["ISIN"])

    out = panel_df.copy()
    out["ISIN"] = out["ISIN"].astype(str).str.strip().str.upper()
    out = out.merge(g, on="ISIN", how="left")
    return out


def map_name_from_us(out_df: pd.DataFrame) -> pd.DataFrame:
    us = pd.read_excel(US_XLSX, usecols=["tic", "conm"])
    us = us.rename(columns={"tic": "Ticker", "conm": "US_Name"})
    us["Ticker"] = us["Ticker"].astype(str).str.strip().str.upper()
    us = us.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"])

    out = out_df.copy()
    out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
    out = out.merge(us, on="Ticker", how="left")
    out["CompanyName"] = out["CompanyName"].fillna(out["US_Name"])
    out = out.drop(columns=["US_Name"])
    return out


def map_name_from_msci(out_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback mapping from MSCI (streaming by chunks) for missing names."""
    out = out_df.copy()
    missing_mask = out["CompanyName"].isna() | (out["CompanyName"].astype(str).str.strip() == "")
    if not missing_mask.any():
        return out

    target_isin = set(out.loc[missing_mask, "ISIN"].astype(str).str.strip().str.upper())
    target_tic = set(out.loc[missing_mask, "Ticker"].astype(str).str.strip().str.upper())

    name_map_isin = {}
    name_map_tic = {}

    usecols = ["ISSUER_ISIN", "ISSUER_TICKER", "ISSUER_NAME"]
    for chunk in pd.read_csv(MSCI_CSV, usecols=usecols, chunksize=200000):
        chunk["ISSUER_ISIN"] = chunk["ISSUER_ISIN"].astype(str).str.strip().str.upper()
        chunk["ISSUER_TICKER"] = chunk["ISSUER_TICKER"].astype(str).str.strip().str.upper()
        chunk["ISSUER_NAME"] = chunk["ISSUER_NAME"].astype(str).str.strip()

        c1 = chunk[chunk["ISSUER_ISIN"].isin(target_isin)]
        for _, r in c1[["ISSUER_ISIN", "ISSUER_NAME"]].dropna().drop_duplicates().iterrows():
            if r["ISSUER_ISIN"] not in name_map_isin and r["ISSUER_NAME"]:
                name_map_isin[r["ISSUER_ISIN"]] = r["ISSUER_NAME"]

        c2 = chunk[chunk["ISSUER_TICKER"].isin(target_tic)]
        for _, r in c2[["ISSUER_TICKER", "ISSUER_NAME"]].dropna().drop_duplicates().iterrows():
            if r["ISSUER_TICKER"] not in name_map_tic and r["ISSUER_NAME"]:
                name_map_tic[r["ISSUER_TICKER"]] = r["ISSUER_NAME"]

        if len(name_map_isin) >= len(target_isin) and len(name_map_tic) >= len(target_tic):
            break

    for idx, row in out[missing_mask].iterrows():
        isin = str(row["ISIN"]).strip().upper()
        tic = str(row["Ticker"]).strip().upper()
        out.at[idx, "CompanyName"] = name_map_isin.get(isin) or name_map_tic.get(tic) or out.at[idx, "CompanyName"]

    return out


def main() -> None:
    panel_df = load_panel_companies()
    out = map_name_from_global(panel_df)
    out = map_name_from_us(out)
    out = map_name_from_msci(out)

    out = out[["ISIN", "CompanyName", "Country", "Ticker"]].drop_duplicates()
    out = out.sort_values(["Country", "CompanyName", "ISIN"], na_position="last").reset_index(drop=True)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    out.to_excel(OUT_XLSX, index=False)

    miss = out["CompanyName"].isna().sum() + (out["CompanyName"].astype(str).str.strip() == "").sum()
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_XLSX}")
    print(f"Rows: {len(out)} | Missing company names: {int(miss)}")


if __name__ == "__main__":
    main()
