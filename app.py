import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import time
import re
from io import BytesIO
from feature_pipeline import extract_features_from_raw

# DB LOGGER IMPORTS
from db_logger import (
    create_run,
    store_raw_file,
    store_portfolio_file,
    update_prediction_time,
    mark_run_failed,
    fetch_all_runs,
    fetch_runs_by_client,
    fetch_raw_file,
    fetch_portfolio_file,
    store_raw_structured_data,
    store_prediction_structured_data
)


# --------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# --------------------------------------------------
st.set_page_config(
    page_title="Propensity-To-Pay Recovery Decision Engine",
    layout="wide"
)

# --------------------------------------------------
# AUTHENTICATION (CENTER SCREEN LOGIN + LOGOUT)
# --------------------------------------------------
def login_page():
    st.markdown("<h2 style='text-align: center;'>🔐 Login</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>Enter your username and password to access the system</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            users = st.secrets["users"]

            if username in users and users[username] == password:
                st.session_state.authenticated = True
                st.session_state.client_name = username

                roles = st.secrets["roles"]
                st.session_state.role = roles.get(username, "client")

                st.success(f"Welcome {username}!")
                st.rerun()

            else:
                st.error("Invalid username or password")


# init session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "client_name" not in st.session_state:
    st.session_state.client_name = None

if "role" not in st.session_state:
    st.session_state.role = "client"


# If not logged in, show login page only
if not st.session_state.authenticated:
    login_page()
    st.stop()

# Logout button (top right)
colA, colB = st.columns([6, 1])
with colB:
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.client_name = None
        st.session_state.role = "client"
        st.rerun()


# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.title("📞 Propensity-To-Pay Recovery Decision Engine")
st.caption("Predict whether a loan should be handled by ICC or referred out")



# --------------------------------------------------
# ADMIN PANEL (ONLY FOR ADMIN USERS)
# --------------------------------------------------
if st.session_state.role == "admin":
    st.sidebar.markdown("---")
    st.sidebar.header("🛠 Admin Panel")

    show_admin = st.sidebar.checkbox("Enable Admin Dashboard")

    if show_admin:
        st.subheader("🛠 Admin Dashboard: Client Test Logs")

        view_mode = st.selectbox(
            "View Mode",
            ["All Runs", "Filter by Client"]
        )

        if view_mode == "Filter by Client":
            client_filter = st.text_input("Enter Client Username")
            if client_filter.strip():
                runs = fetch_runs_by_client(client_filter.strip())
            else:
                runs = []
        else:
            runs = fetch_all_runs()

        if not runs:
            st.info("No runs found.")
        else:
            df_runs = pd.DataFrame(runs)
            st.dataframe(df_runs, use_container_width=True)

            run_ids = df_runs["run_id"].tolist()
            selected_run = st.selectbox("Select run_id to download files", run_ids)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Download Raw File"):
                    row = fetch_raw_file(selected_run)
                    if row is None:
                        st.error("Raw file not found in DB.")
                    else:
                        file_name, file_data = row
                        st.download_button(
                            label="⬇️ Download Raw File",
                            data=file_data,
                            file_name=file_name,
                            mime="application/octet-stream"
                        )

            with col2:
                if st.button("Download Portfolio File"):
                    row = fetch_portfolio_file(selected_run)
                    if row is None:
                        st.error("Portfolio file not found in DB.")
                    else:
                        file_name, file_data = row
                        st.download_button(
                            label="⬇️ Download Portfolio File",
                            data=file_data,
                            file_name=file_name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

        st.stop()


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

THRESHOLD = 0.4

DECISION_LABELS = {1: "ICC RECOVERABLE", 0: "REFER OUT"}
DECISION_COLORS = {1: "green", 0: "red"}

base_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(base_dir, "..", "outputs", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "batch_file_ready" not in st.session_state:
    st.session_state.batch_file_ready = False

if "batch_file_path" not in st.session_state:
    st.session_state.batch_file_path = None

if "run_id" not in st.session_state:
    st.session_state.run_id = None

if "portfolio_excel" not in st.session_state:
    st.session_state.portfolio_excel = None

if "portfolio_filename" not in st.session_state:
    st.session_state.portfolio_filename = None

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "outputs", "models", "final_model_2.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    return joblib.load(model_path)

model = load_model()

# --------------------------------------------------
# EXPORT HELPERS
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_excel(dataframe):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Decisions")
    buffer.seek(0)
    return buffer

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

file_mode = st.sidebar.radio(
    "Upload File Type",
    ["Raw File", "Preprocessed Feature File"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Monthly File (Excel or CSV)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("👈 Upload a monthly file (raw or preprocessed) to begin")
    st.stop()

# --------------------------------------------------
# CREATE RUN + STORE RAW FILE IN DB
# --------------------------------------------------
try:
    client_name = st.session_state.client_name
    raw_file_bytes = uploaded_file.getvalue()

    run_id = create_run(
        client_name=client_name,
        raw_file_name=uploaded_file.name,
        threshold_used=THRESHOLD
    )

    # store_raw_file(run_id, uploaded_file.name, raw_file_bytes)
    st.session_state.run_id = run_id

except Exception as e:
    st.error(f"Database logging failed (Raw File Store): {e}")
    st.stop()

# --------------------------------------------------
# LOAD FILE
# --------------------------------------------------
if uploaded_file.name.endswith(".csv"):
    df_raw = pd.read_csv(uploaded_file)

else:
    if file_mode == "Raw File":

        df_raw = pd.read_excel(uploaded_file, header=0, engine = "openpyxl")

        cols_lower = [str(c).strip().lower() for c in df_raw.columns]
        expected_cols = {"loan_id", "pos", "emi amount"}

        if not expected_cols.issubset(set(cols_lower)):
            df_raw = pd.read_excel(uploaded_file, header=1, engine = "openpyxl")

    else:
        df_raw = pd.read_excel(uploaded_file, engine = "openpyxl")

# --------------------------------------------------
# FEATURE EXTRACTION / VALIDATION
# --------------------------------------------------
if file_mode == "Raw File":
    try:
        df = extract_features_from_raw(df_raw)
    except Exception as e:
        mark_run_failed(run_id, str(e))
        st.error(f"Feature extraction failed: {e}")
        st.stop()
else:
    df = df_raw.copy()

    required_cols = set(FEATURE_COLS + [LOAN_ID_COL])
    missing = required_cols - set(df.columns)

    if missing:
        mark_run_failed(run_id, f"Missing columns in preprocessed file: {missing}")
        st.error(f"Missing required columns in preprocessed file: {missing}")
        st.stop()

# --------------------------------------------------
# PREDICT
# --------------------------------------------------
try:
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)

    df["icc_recovery_probability"] = model.predict_proba(X)[:, 1]
    df["decision"] = (df["icc_recovery_probability"] >= THRESHOLD).astype(int)
    df["decision_label"] = df["decision"].map(DECISION_LABELS)


    # -------------------------------
    # STORE STRUCTURED DATA IN TABLES
    # -------------------------------
    with st.spinner("🗄️ Storing structured raw data..."):
        store_raw_structured_data(run_id, df )

    with st.spinner("🗄️ Storing structured prediction data..."):
        store_prediction_structured_data(run_id, df)

except Exception as e:
    mark_run_failed(run_id, f"Prediction failed: {e}")
    st.error(f"Prediction failed: {e}")
    st.stop()

# --------------------------------------------------
# STORE PORTFOLIO FILE IN DB IMMEDIATELY
# --------------------------------------------------
try:
    portfolio_excel = generate_excel(df)
    portfolio_filename = f"{os.path.splitext(uploaded_file.name)[0]}_portfolio_run_{run_id}.xlsx"

    store_portfolio_file(run_id, portfolio_filename, portfolio_excel.getvalue())
    update_prediction_time(run_id, portfolio_filename)

    st.session_state.portfolio_excel = portfolio_excel
    st.session_state.portfolio_filename = portfolio_filename

except Exception as e:
    mark_run_failed(run_id, f"Portfolio storage failed: {e}")
    st.error(f"Database logging failed (Portfolio Store): {e}")
    st.stop()

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

        pattern = "|".join([re.escape(x) for x in loan_list])

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
# EXPORT SECTION
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("📤 Export Decisions")

if st.sidebar.button("Generate Full Portfolio File"):

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    eta_text = st.sidebar.empty()

    start_time = time.time()

    def update_progress(pct, message):
        elapsed = time.time() - start_time
        eta = int((elapsed / pct) * (100 - pct)) if pct > 0 else 0

        progress_bar.progress(pct)
        status_text.markdown(f"**{message} ({pct}%)**")
        eta_text.markdown(f"⏳ ETA: ~{eta} sec" if pct < 100 else " ")

    for i in range(1, 31):
        update_progress(i, "🔄 Initializing export")
        time.sleep(0.02)

    for i in range(31, 71):
        update_progress(i, "📊 Processing loan data")
        time.sleep(0.02)

    update_progress(71, "📝 Writing Excel file")

    for i in range(72, 100):
        update_progress(i, "📝 Finalizing file")
        time.sleep(0.02)

    progress_bar.progress(100)
    status_text.markdown("### 🎉 File ready (100%)")
    eta_text.empty()

    st.sidebar.download_button(
        label="⬇️ Download Excel",
        data=st.session_state.portfolio_excel,
        file_name=st.session_state.portfolio_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
