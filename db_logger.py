import mysql.connector
from mysql.connector import Error
from datetime import datetime
import streamlit as st


# --------------------------------------------------
# DB CONFIG
# --------------------------------------------------
DB_CONFIG = {
    "host": st.secrets["DB_HOST"],
    "user": st.secrets["DB_USER"],
    "password": st.secrets["DB_PASSWORD"],
    "database": st.secrets["DB_NAME"],
    "port": st.secrets["DB_PORT"],
}


# --------------------------------------------------
# CONNECTION HELPER
# --------------------------------------------------
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


# --------------------------------------------------
# RUN MANAGEMENT
# --------------------------------------------------
def create_run(client_name, raw_file_name, threshold_used=0.4):
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO test_runs
                (client_name, raw_file_name, threshold_used, status, upload_time)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(
                    query,
                    (client_name, raw_file_name, threshold_used, "SUCCESS", datetime.now())
                )
                conn.commit()
                return cursor.lastrowid
    except Error as e:
        raise Exception(f"Create run failed: {e}")


def update_prediction_time(run_id, portfolio_file_name):
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                UPDATE test_runs
                SET prediction_time = %s,
                    portfolio_file_name = %s
                WHERE run_id = %s
                """
                cursor.execute(
                    query,
                    (datetime.now(), portfolio_file_name, run_id)
                )
                conn.commit()
    except Error as e:
        raise Exception(f"Update prediction time failed: {e}")


def mark_run_failed(run_id, error_message):
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                UPDATE test_runs
                SET status = %s,
                    error_message = %s
                WHERE run_id = %s
                """
                cursor.execute(
                    query,
                    ("FAILED", error_message[:500], run_id)
                )
                conn.commit()
    except Error as e:
        raise Exception(f"Mark run failed error: {e}")


# --------------------------------------------------
# BLOB STORAGE
# --------------------------------------------------
def store_raw_file(run_id, file_name, file_bytes):
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO raw_files (run_id, file_name, file_data)
                VALUES (%s, %s, %s)
                """
                cursor.execute(query, (run_id, file_name, file_bytes))
                conn.commit()
    except Error as e:
        raise Exception(f"Store raw file failed: {e}")


def store_portfolio_file(run_id, file_name, file_bytes):
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO portfolio_files (run_id, file_name, file_data)
                VALUES (%s, %s, %s)
                """
                cursor.execute(query, (run_id, file_name, file_bytes))
                conn.commit()
    except Error as e:
        raise Exception(f"Store portfolio file failed: {e}")


# --------------------------------------------------
# STRUCTURED RAW DATA STORAGE (FAST BATCH INSERT)
# --------------------------------------------------
def store_raw_structured_data(run_id, df_raw):

    if df_raw.empty:
        return

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:

                insert_query = """
                INSERT INTO raw_uploaded_data
                (run_id, loan_id, emi_amount, pos, bucket_raw,
                 calling_attempts, connected, ptp_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """

                # Faster than iterrows
                data_to_insert = [
                    (
                        run_id,
                        str(row.loan_id),
                        float(row.emi_amount),
                        float(row.pos),
                        str(getattr(row, "bucket_raw", "")),
                        int(row.calling_attempts),
                        int(row.connected),
                        int(row.ptp_count)
                    )
                    for row in df_raw.itertuples(index=False)
                ]

                cursor.executemany(insert_query, data_to_insert)
                conn.commit()

    except Error as e:
        raise Exception(f"Store structured raw data failed: {e}")


# --------------------------------------------------
# STRUCTURED PREDICTION STORAGE (FAST BATCH INSERT)
# --------------------------------------------------
def store_prediction_structured_data(run_id, df_pred):

    if df_pred.empty:
        return

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:

                insert_query = """
                INSERT INTO prediction_data
                (run_id, loan_id, icc_recovery_probability, decision, decision_label)
                VALUES (%s, %s, %s, %s, %s)
                """

                data_to_insert = [
                    (
                        run_id,
                        str(row.loan_id),
                        float(row.icc_recovery_probability),
                        int(row.decision),
                        str(row.decision_label)
                    )
                    for row in df_pred.itertuples(index=False)
                ]

                cursor.executemany(insert_query, data_to_insert)
                conn.commit()

    except Error as e:
        raise Exception(f"Store structured prediction data failed: {e}")


# --------------------------------------------------
# FETCH FUNCTIONS (ADMIN PANEL)
# --------------------------------------------------
def fetch_all_runs():
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT * FROM test_runs ORDER BY run_id DESC")
            return cursor.fetchall()


def fetch_runs_by_client(client_name):
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cursor:
            query = """
            SELECT * FROM test_runs
            WHERE client_name = %s
            ORDER BY run_id DESC
            """
            cursor.execute(query, (client_name,))
            return cursor.fetchall()


def fetch_raw_file(run_id):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT file_name, file_data FROM raw_files WHERE run_id=%s",
                (run_id,)
            )
            return cursor.fetchone()


def fetch_portfolio_file(run_id):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT file_name, file_data FROM portfolio_files WHERE run_id=%s",
                (run_id,)
            )

            return cursor.fetchone()
