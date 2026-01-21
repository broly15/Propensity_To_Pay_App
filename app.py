import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Propensity-To-Pay Recovery Decision Engine",
    layout="wide"
)

st.title("📞 Propensity-To-Pay Recovery Decision Engine")
st.caption("Predict whether a loan should be handled by ICC or referred out")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
LOAN_ID_COL = "loan_id"

FEATURE_COLS = [
    "emi_amount",
    "pos",
    "bucket_numeric",
    "calling_attempts",
    "connected",
    "ptp_count"
]

THRESHOLD = 0.5

DECISION_LABELS = {1: "ICC RECOVERABLE", 0: "REFER OUT"}
DECISION_COLORS = {1: "green", 0: "red"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# SESSION STATE (CRITICAL FIX)
# --------------------------------------------------
if "batch_file_ready" not in st.session_state:
    st.session_state.batch_file_ready = False

if "batch_file_path" not in st.session_state:
    st.session_state.batch_file_path = None

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir,"outputs" , "models", "final_model.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    return joblib.load(model_path)

model = load_model()


# --------------------------------------------------
# VISUAL HELPERS
# --------------------------------------------------
def plot_distribution_pie(data, title):
    summary = (
        data.groupby("decision_label")
        .size()
        .reindex(["ICC RECOVERABLE", "REFER OUT"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(
        summary,
        labels=summary.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[DECISION_COLORS[1], DECISION_COLORS[0]]
    )
    ax.axis("equal")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


def plot_single_decision(decision_label):
    color = DECISION_COLORS[1] if decision_label == "ICC RECOVERABLE" else DECISION_COLORS[0]
    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    ax.pie([1], colors=[color])
    ax.axis("equal")
    ax.set_title(decision_label)
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------
st.sidebar.header("📂 Upload Monthly Loan File")

uploaded_file = st.sidebar.file_uploader(
    "Upload Processed Monthly File (Excel or CSV)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("👈 Upload a processed monthly file to begin")
    st.stop()

# --------------------------------------------------
# LOAD + PREDICT
# --------------------------------------------------
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

required_cols = set(FEATURE_COLS + [LOAN_ID_COL])
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)

df["icc_recovery_probability"] = model.predict_proba(X)[:, 1]
df["decision"] = (df["icc_recovery_probability"] >= THRESHOLD).astype(int)
df["decision_label"] = df["decision"].map(DECISION_LABELS)

st.success(f"Loaded and processed {len(df)} loans")

# --------------------------------------------------
# SIDEBAR — SELECTION MODE
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔍 Loan Selection")

selection_mode = st.sidebar.radio(
    "Selection Mode",
    ["Drilldown by Decision", "Single Loan Search", "Bulk Loan Search"]
)

# --------------------------------------------------
# DRILLDOWN MODE
# --------------------------------------------------
if selection_mode == "Drilldown by Decision":
    icc_count = (df["decision_label"] == "ICC RECOVERABLE").sum()
    refer_count = (df["decision_label"] == "REFER OUT").sum()

    k1, k2 = st.columns(2)
    with k1:
        st.metric("🟢 ICC Recoverable Loans", f"{icc_count:,}")
    with k2:
        st.metric("🔴 Refer Out Loans", f"{refer_count:,}")

    plot_distribution_pie(df, "Overall Portfolio")

    def paginated_table(data, key):
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=max(1, (len(data) - 1) // 10 + 1),
            step=1,
            key=key
        )
        start = (page - 1) * 10
        end = start + 10
        st.dataframe(data.iloc[start:end])

    st.subheader("🟢 ICC Recoverable Loans")
    paginated_table(df[df["decision_label"] == "ICC RECOVERABLE"], "icc_page")

    st.subheader("🔴 Refer Out Loans")
    paginated_table(df[df["decision_label"] == "REFER OUT"], "refer_page")

# --------------------------------------------------
# SINGLE LOAN SEARCH
# --------------------------------------------------
elif selection_mode == "Single Loan Search":
    loan_input = st.sidebar.text_input("Enter Loan Number (partial allowed)")

    if loan_input.strip():
        customer = df[
            df[LOAN_ID_COL]
            .astype(str)
            .str.contains(loan_input.strip(), case=False, na=False)
        ]

        if customer.empty:
            st.error("No matching loan found")
            st.stop()

        row = customer.iloc[0]

        c1, c2 = st.columns([1.2, 2.8])
        with c1:
            plot_single_decision(row["decision_label"])
        with c2:
            prob = row["icc_recovery_probability"] * 100
            emoji = "🔥" if prob >= 70 else "⚠️" if prob >= 50 else "❄️"

            st.metric(
                label="📊 ICC Recovery Probability",
                value=f"{prob:.1f}%",
                delta=f"{emoji} {'High' if prob>=70 else 'Medium' if prob>=50 else 'Low'} Confidence"
            )

        st.subheader("📄 Loan Details")
        st.dataframe(customer)

# --------------------------------------------------
# BULK LOAN SEARCH
# --------------------------------------------------
else:
    loan_input = st.sidebar.text_area(
        "Enter Loan Numbers (comma / newline separated, partial allowed)"
    )

    if loan_input.strip():
        loan_list = [
            x.strip()
            for x in loan_input.replace(",", "\n").split("\n")
            if x.strip()
        ]

        pattern = "|".join(loan_list)

        customer_df = df[
            df[LOAN_ID_COL]
            .astype(str)
            .str.contains(pattern, case=False, na=False)
        ]

        if customer_df.empty:
            st.error("No matching loans found")
            st.stop()

        if len(customer_df) == 1:
            row = customer_df.iloc[0]
            c1, c2 = st.columns([1.2, 2.8])
            with c1:
                plot_single_decision(row["decision_label"])
            with c2:
                prob = row["icc_recovery_probability"] * 100
                emoji = "🔥" if prob >= 70 else "⚠️" if prob >= 50 else "❄️"

                st.metric(
                    label="📊 ICC Recovery Probability",
                    value=f"{prob:.1f}%",
                    delta=f"{emoji} {'High' if prob>=70 else 'Medium' if prob>=50 else 'Low'} Confidence"
                )
        else:
            plot_distribution_pie(customer_df, "Selected Loans Decision Split")

        st.subheader("📄 Selected Loan Details")
        st.dataframe(customer_df)

# --------------------------------------------------
# EXPORT (CLOUD-SAFE | NO CRASH)
# --------------------------------------------------
from io import BytesIO

st.sidebar.markdown("---")
st.sidebar.header("📤 Export Decisions")

@st.cache_data(show_spinner=False)
def generate_excel(dataframe):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Decisions")
    buffer.seek(0)
    return buffer

if st.sidebar.button("Generate Full Portfolio File"):
    with st.spinner("Preparing Excel file..."):
        excel_file = generate_excel(df)

    st.sidebar.success("File ready for download")

    st.sidebar.download_button(
        label="⬇️ Download Excel",
        data=excel_file,
        file_name="icc_decisions_full_portfolio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
