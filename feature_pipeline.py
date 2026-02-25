import pandas as pd
import re

# -----------------------------
# CONFIG
# -----------------------------
REQUIRED_RAW_COLS = [
    "loan_id",
    "emi amount",
    "pos",
    "bucket",
    "calling attempt by icc",
    "connected",
    "count of ptp"
]

RENAME_MAP = {
    "loan_id": "loan_id",
    "emi amount": "emi_amount",
    "pos": "pos",
    "bucket": "bucket_raw",
    "calling attempt by icc": "calling_attempts",
    "connected": "connected",
    "count of ptp": "ptp_count"
}

BUCKET_MAP = {
    "current": 0,
    "0": 0,
    "1-30": 15,
    "31-60": 45,
    "61-90": 75,
    "90+": 120,
    "90 +": 120,
    "90plus": 120,
    "90 plus": 120
}

NUMERIC_COLS = [
    "emi_amount",
    "pos",
    "bucket_numeric",
    "calling_attempts",
    "connected",
    "ptp_count"
]


# -----------------------------
# HELPERS
# -----------------------------
def normalize_col(col: str) -> str:
    """
    Normalize column names:
    - lowercase
    - strip
    - replace multiple spaces with single space
    """
    col = str(col).strip().lower()
    col = re.sub(r"\s+", " ", col)
    return col


def normalize_bucket(val) -> str:
    """
    Normalize bucket values to improve mapping accuracy.
    """
    if pd.isna(val):
        return "current"

    val = str(val).strip().lower()
    val = re.sub(r"\s+", " ", val)

    # normalize common variants
    val = val.replace("days", "").strip()
    val = val.replace(" ", "")

    # now val becomes like "1-30", "90+"
    return val


# -----------------------------
# MAIN FEATURE PIPELINE
# -----------------------------
def extract_features_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw uploaded monthly loan file into model-ready feature dataframe.
    Returns dataframe containing:
    loan_id + feature columns required for prediction.
    """

    if df_raw is None or df_raw.empty:
        raise ValueError("Uploaded file is empty.")

    df_raw = df_raw.copy()

    # Normalize column names
    df_raw.columns = [normalize_col(c) for c in df_raw.columns]

    # Validate required columns
    missing_cols = set(REQUIRED_RAW_COLS) - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in raw file: {sorted(missing_cols)}")

    # Keep only needed cols (saves memory)
    df = df_raw[REQUIRED_RAW_COLS].copy()

    # Rename cols
    df.rename(columns=RENAME_MAP, inplace=True)

    # Bucket numeric conversion
    df["bucket_clean"] = df["bucket_raw"].apply(normalize_bucket)
    df["bucket_numeric"] = df["bucket_clean"].map(BUCKET_MAP).fillna(0)

    # Convert numeric cols in one shot
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Optimize dtypes for performance/memory
    df["connected"] = df["connected"].astype("int16", errors="ignore")
    df["calling_attempts"] = df["calling_attempts"].astype("int32", errors="ignore")
    df["ptp_count"] = df["ptp_count"].astype("int16", errors="ignore")
    df["bucket_numeric"] = df["bucket_numeric"].astype("int16", errors="ignore")

    

    return df
