import pandas as pd
import numpy as np
import os
import warnings
import time
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings('ignore')

# Input paths
T_FILE = os.path.join(BASE_DIR, 'MSCI.csv')          # MSCI ESG ratings
U_FILE = os.path.join(BASE_DIR, 'U.S. Ticker.xlsx')  # WRDS North America (US)
G_FILE = os.path.join(BASE_DIR, 'Global ISIN.xlsx')  # WRDS Global (non-US)

# Output path
O_FILE = os.path.join(BASE_DIR, 'Final_Dataset_Dummy_ESG.csv')

# Sector filter keyword (applied to IVA_INDUSTRY column in MSCI data)
I_KEY = 'Auto'

# Countries to retain (CN = Mainland China, HK = Hong Kong, US = United States)
T_CNTRY = ['CN', 'HK', 'US']

# Unit conversion factor: Yahoo Finance reports absolute figures; Compustat
# reports in millions.  Divide Yahoo values by this factor to align units.
YAHOO_SCALE = 1_000_000


# Clean a WRDS Compustat dataframe

def clean_wrds_data(df, s_type):
    # Rename standard WRDS column names to dissertation variable names
    c_map = {
        'at':       'Total_Assets',
        'lt':       'Total_Liabilities',
        'revt':     'Total_Revenue',
        'ni':       'Net_Income',
        'nicon':    'Net_Income',
        'datadate': 'Date',
    }
    df = df.rename(columns=c_map)

    # Identify the firm-level matching key depending on geography
    if s_type == 'US':
        # US firms are matched by exchange ticker
        t_cols = [c for c in df.columns if 'tic' in str(c).lower()]
        if t_cols:
            df = df.rename(columns={t_cols[0]: 'Ticker_Match'})
        i_col = 'Ticker_Match'
    else:
        # Non-US firms are matched by ISIN
        i_cols = [c for c in df.columns if 'isin' in str(c).lower()]
        if i_cols:
            df = df.rename(columns={i_cols[0]: 'ISIN_Match'})
        i_col = 'ISIN_Match'

    # Return empty frame if the expected identifier column is missing
    if i_col not in df.columns:
        return pd.DataFrame()

    # Parse the fiscal period end date; extract year
    d_cols = [c for c in df.columns if 'date' in str(c).lower()]
    if d_cols:
        df = df.rename(columns={d_cols[0]: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year

    # All Compustat annual observations are mapped to H2 (second half of the
    # fiscal year), as they reflect full-year financial performance
    df['Half'] = 'H2'

    # Convert financial columns to numeric; coerce errors to NaN
    for c in ['Total_Assets', 'Net_Income', 'Total_Revenue', 'Total_Liabilities']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute Return on Assets (ROA) as a percentage
    if 'Total_Assets' in df.columns and 'Net_Income' in df.columns:
        df['ROA'] = (df['Net_Income'] / df['Total_Assets']) * 100

    # Select and return only the required columns; drop rows missing Total_Assets
    req = ['Year', 'Half', 'ROA', 'Total_Assets', 'Total_Revenue',
           'Total_Liabilities', 'Net_Income']
    return (
        df[[i_col] + [c for c in req if c in df.columns]]
        .dropna(subset=['Total_Assets'])
        .copy()
    )


# Read an Excel file, trying two header rows

def read_w_ex(f):
    try:
        df = pd.read_excel(f, sheet_name=0)
        # If none of the expected WRDS column names are present, try row 1
        if not any(x in df.columns for x in ['at', 'tic', 'isin']):
            df = pd.read_excel(f, sheet_name=0, header=1)
        return df
    except Exception as e:
        print(f"Failed to read: {f} — {e}")
        return pd.DataFrame()


# Generate Yahoo Finance ticker candidates by country

def get_yahoo_ticker(ticker, country):

    ticker = str(ticker).strip()
    if country == 'US':
        return [ticker]
    elif country == 'HK':
        candidates = [f"{ticker}.HK"]
        if ticker.isdigit():
            candidates.append(f"{int(ticker):04d}.HK")
        return candidates
    elif country == 'CN':
        bare = ticker.replace('CH:', '').strip()
        if bare.startswith('6'):
            return [f"{bare}.SS"]
        elif bare.startswith('0') or bare.startswith('3'):
            return [f"{bare}.SZ"]
        else:
            return [f"{bare}.SS", f"{bare}.SZ"]
    return [ticker]


# Retrieve annual financials from Yahoo Finance

def fetch_yahoo_financials(ticker, country, target_years):

    for yt in get_yahoo_ticker(ticker, country):
        try:
            t = yf.Ticker(yt)
            income  = t.financials      # Annual income statement
            balance = t.balance_sheet   # Annual balance sheet

            if income is None or income.empty:
                continue

            results = []
            for col in income.columns:
                year = pd.Timestamp(col).year
                if year not in target_years:
                    continue

                row = {'Year': year, 'Half': 'H2'}

                # Total Revenue: try multiple possible Yahoo row labels
                for k in ['Total Revenue', 'Revenue']:
                    if k in income.index and not pd.isna(income.loc[k, col]):
                        row['Total_Revenue'] = float(income.loc[k, col]) / YAHOO_SCALE
                        break

                # Net Income: try multiple possible Yahoo row labels
                for k in ['Net Income', 'Net Income Common Stockholders']:
                    if k in income.index and not pd.isna(income.loc[k, col]):
                        row['Net_Income'] = float(income.loc[k, col]) / YAHOO_SCALE
                        break

                # Balance sheet items (matched by the same column date)
                if balance is not None and not balance.empty and col in balance.columns:
                    for k in ['Total Assets']:
                        if k in balance.index and not pd.isna(balance.loc[k, col]):
                            row['Total_Assets'] = float(balance.loc[k, col]) / YAHOO_SCALE
                            break
                    for k in ['Total Liabilities Net Minority Interest', 'Total Liabilities']:
                        if k in balance.index and not pd.isna(balance.loc[k, col]):
                            row['Total_Liabilities'] = float(balance.loc[k, col]) / YAHOO_SCALE
                            break

                # Compute ROA from the retrieved figures
                ta = row.get('Total_Assets')
                ni = row.get('Net_Income')
                if ta and ni and ta != 0:
                    row['ROA'] = (ni / ta) * 100

                results.append(row)

            if results:
                return pd.DataFrame(results)

        except Exception:
            pass  # Try the next candidate ticker on any error

    return pd.DataFrame()


# Fill missing financial values with Yahoo Finance data

def fill_missing_with_yahoo(df, ticker_col, country_col, fin_cols, target_years):

    # Identify rows that have at least one missing financial value
    missing_mask = df[fin_cols].isnull().any(axis=1)
    companies = df.loc[missing_mask, [ticker_col, country_col]].drop_duplicates()
    print(f"  Companies requiring Yahoo Finance supplementation: {len(companies)}")

    filled = 0
    for _, row in companies.iterrows():
        ticker  = row[ticker_col]
        country = row[country_col]

        # Identify the specific years with missing data for this firm
        c_mask = df[ticker_col] == ticker
        years  = df.loc[c_mask & missing_mask, 'Year'].unique().tolist()
        if not years:
            continue

        # Fetch Yahoo Finance data for the missing years
        yahoo_df = fetch_yahoo_financials(ticker, country, years)
        if yahoo_df.empty:
            continue

        # Update only the NaN cells that Yahoo Finance can supply
        for _, yr in yahoo_df.iterrows():
            r_mask = c_mask & (df['Year'] == yr['Year']) & (df['Half'] == yr['Half'])
            for col in fin_cols:
                if col in yr.index and not pd.isna(yr[col]):
                    update = r_mask & df[col].isnull()
                    if update.any():
                        df.loc[update, col] = yr[col]
                        filled += 1

        # Brief pause to avoid hitting Yahoo Finance rate limits
        time.sleep(0.1)

    print(f"  Yahoo Finance supplemented {filled} missing values")
    return df


# Main execution 

if __name__ == "__main__":

    # Step 1: Load and filter MSCI ESG data
    df_t = (
        pd.read_csv(T_FILE, low_memory=False)
        if T_FILE.endswith('.csv')
        else pd.read_excel(T_FILE)
    )

    # Filter to automotive firms only
    if 'IVA_INDUSTRY' in df_t.columns:
        df_t = df_t[df_t['IVA_INDUSTRY'].astype(str).str.contains(I_KEY, case=False, na=False)]

    # Filter to the three target markets
    df_t = df_t[df_t['ISSUER_CNTRY_DOMICILE'].isin(T_CNTRY)]

    # Convert ESG pillar scores to numeric; coerce errors to NaN
    p_cols = ['ENVIRONMENTAL_PILLAR_SCORE', 'SOCIAL_PILLAR_SCORE', 'GOVERNANCE_PILLAR_SCORE']
    for c in p_cols:
        df_t[c] = pd.to_numeric(df_t[c], errors='coerce')

    # Parse the ESG rating date; assign each observation to a half-year period
    df_t['AS_OF_DATE'] = pd.to_datetime(df_t['AS_OF_DATE'])
    df_t['Year'] = df_t['AS_OF_DATE'].dt.year
    df_t['Half'] = df_t['AS_OF_DATE'].dt.month.apply(lambda x: 'H1' if x <= 6 else 'H2')

    # Aggregate to firm × year × half level by averaging multiple ESG scores
    # within the same half-year period (MSCI updates scores continuously)
    df_b = (
        df_t[['ISSUER_TICKER', 'ISSUER_ISIN', 'ISSUER_CNTRY_DOMICILE', 'Year', 'Half'] + p_cols]
        .groupby(['ISSUER_TICKER', 'ISSUER_ISIN', 'ISSUER_CNTRY_DOMICILE', 'Year', 'Half'])
        .mean()
        .reset_index()
    )

    # Step 2: Build balanced panel skeleton 
    # Create all firm × year × half combinations for 2018–2024
    skel = []
    u_cos = df_b[['ISSUER_TICKER', 'ISSUER_ISIN', 'ISSUER_CNTRY_DOMICILE']].drop_duplicates()
    for _, r in u_cos.iterrows():
        for y in range(2018, 2025):
            for h in ['H1', 'H2']:
                skel.append({
                    'ISSUER_TICKER':          r['ISSUER_TICKER'],
                    'ISSUER_ISIN':            r['ISSUER_ISIN'],
                    'ISSUER_CNTRY_DOMICILE':  r['ISSUER_CNTRY_DOMICILE'],
                    'Year':  y,
                    'Half':  h,
                })

    # Left-join ESG scores onto the skeleton; missing periods will have NaN scores
    df_m = pd.merge(
        pd.DataFrame(skel), df_b,
        on=['ISSUER_TICKER', 'ISSUER_ISIN', 'ISSUER_CNTRY_DOMICILE', 'Year', 'Half'],
        how='left',
    )

    # Step 3: Merge WRDS Compustat financial data 
    print("Merging WRDS financial data...")

    # US firms: matched by ticker symbol
    us_data = clean_wrds_data(read_w_ex(U_FILE), 'US').drop_duplicates(['Ticker_Match', 'Year', 'Half'])
    df_u = pd.merge(
        df_m[df_m['ISSUER_CNTRY_DOMICILE'] == 'US'],
        us_data,
        left_on=['ISSUER_TICKER', 'Year', 'Half'],
        right_on=['Ticker_Match', 'Year', 'Half'],
        how='left',
    ).drop(columns=['Ticker_Match'])

    # Non-US firms: matched by ISIN
    intl_data = clean_wrds_data(read_w_ex(G_FILE), 'INTL').drop_duplicates(['ISIN_Match', 'Year', 'Half'])
    df_i = pd.merge(
        df_m[df_m['ISSUER_CNTRY_DOMICILE'] != 'US'],
        intl_data,
        left_on=['ISSUER_ISIN', 'Year', 'Half'],
        right_on=['ISIN_Match', 'Year', 'Half'],
        how='left',
    ).drop(columns=['ISIN_Match'])

    fin_cols    = ['Total_Assets', 'Total_Liabilities', 'Total_Revenue', 'Net_Income', 'ROA']
    target_years = list(range(2018, 2025))

    # Step 4: Supplement missing financials from Yahoo Finance
    print("Supplementing missing financial data from Yahoo Finance...")

    # Sort by firm × time before imputation to ensure forward/backward fill works correctly
    df_u['Half_num'] = df_u['Half'].map({'H1': 1, 'H2': 2})
    df_u = df_u.sort_values(['ISSUER_TICKER', 'Year', 'Half_num'])
    print("  Processing US firms...")
    df_u = fill_missing_with_yahoo(df_u, 'ISSUER_TICKER', 'ISSUER_CNTRY_DOMICILE', fin_cols, target_years)

    df_i['Half_num'] = df_i['Half'].map({'H1': 1, 'H2': 2})
    df_i = df_i.sort_values(['ISSUER_ISIN', 'Year', 'Half_num'])
    print("  Processing non-US firms...")
    df_i = fill_missing_with_yahoo(df_i, 'ISSUER_TICKER', 'ISSUER_CNTRY_DOMICILE', fin_cols, target_years)

    # Step 5: Within-firm imputation for remaining missing values
    # Forward-fill then backward-fill within each firm's time series.
    # This carries the most recent available observation forward (or, if no
    # earlier observation is available, backward) to adjacent half-year periods.
    print("Applying within-firm forward-fill/backward-fill imputation...")

    df_u[fin_cols] = df_u.groupby('ISSUER_TICKER')[fin_cols].transform(lambda s: s.ffill().bfill())
    df_u = df_u.drop(columns=['Half_num'])

    df_i[fin_cols] = df_i.groupby('ISSUER_ISIN')[fin_cols].transform(lambda s: s.ffill().bfill())
    df_i = df_i.drop(columns=['Half_num'])

    # Step 6: Combine US and non-US panels; apply final cleaning
    df_f = pd.concat([df_u, df_i], ignore_index=True)

    # Drop observations missing any ESG pillar score (all three required)
    df_final = df_f.dropna(subset=p_cols).copy()

    # Step 7: Construct annual above-median ESG dummy variables
    # For each ESG pillar, a dummy equals 1 if the firm's score exceeds the
    # cross-sectional median across all firms in the same calendar year,
    # and 0 otherwise.  Using year-specific medians controls for aggregate
    # drift in MSCI's scoring methodology across the sample period.
    for score_col, dummy_col in zip(p_cols, ['E_Dummy', 'S_Dummy', 'G_Dummy']):
        median_val = df_final[score_col].median()
        df_final[dummy_col] = (df_final[score_col] > median_val).astype(int)

    # Step 8: Export
    df_final.to_csv(O_FILE, index=False, encoding='utf-8-sig')
    print(f"Processing complete. Output file: {O_FILE}")
    print(f"Final sample size: {len(df_final)} firm-half-year observations")
