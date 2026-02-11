import pandas as pd


def extract_features_from_raw(df_raw):
    """
    Converts raw uploaded monthly loan file into model-ready feature dataframe.
    Returns dataframe containing:
    loan_id + FEATURE_COLS required for prediction.
    """

    df_raw = df_raw.copy()

    # --- Standardize column names (IMPORTANT) ---
    df_raw.columns = df_raw.columns.astype(str).str.strip().str.lower()

    # --- Required raw columns (must be lowercase now) ---
    KEEP_COLS = [
        "loan_id",
        "emi amount",
        "pos",
        "bucket",
        "calling attempt  by icc",
        "connected",
        "count of ptp"
    ]

    # --- Rename mapping (lowercase keys only) ---
    RENAME_MAP = {
        "loan_id": "loan_id",
        "emi amount": "emi_amount",
        "pos": "pos",
        "bucket": "bucket_raw",
        "calling attempt  by icc": "calling_attempts",
        "connected": "connected",
        "count of ptp": "ptp_count"
    }

    # --- Bucket conversion mapping ---
    bucket_map = {
        "current": 0,
        "1-30": 15,
        "31-60": 45,
        "61-90": 75,
        "90+": 120
    }

    # --- Validate required columns ---
    missing_cols = set(KEEP_COLS) - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in raw file: {missing_cols}")

    # --- Keep only required columns ---
    df = df_raw[KEEP_COLS].copy()

    # --- Rename columns ---
    df.rename(columns=RENAME_MAP, inplace=True)

    # --- Bucket numeric ---
    df["bucket_raw"] = df["bucket_raw"].astype(str).str.strip().str.lower()
    df["bucket_numeric"] = df["bucket_raw"].map(bucket_map).fillna(0)

    # --- Convert numeric columns ---
    NUMERIC_COLS = [
        "emi_amount",
        "pos",
        "bucket_numeric",
        "calling_attempts",
        "connected",
        "ptp_count"
    ]

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # remove bucket_raw (not needed in UI)
    df.drop(columns=["bucket_raw"], inplace=True)

    return df

