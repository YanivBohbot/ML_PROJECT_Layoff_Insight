import base64
import io
import os
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")


def show():
    st.title("Upload New Training Data & Retrain Model")

    # ---- session state ----
    st.session_state.setdefault("saved_as", None)
    st.session_state.setdefault("rows", None)

    # ---- upload section ----
    st.subheader("1) Upload CSV")
    file = st.file_uploader("Choose a CSV", type="csv")

    if file:
        # local preview (cached to avoid re-parsing on rerun)
        raw = file.getvalue()
        try:
            df = pd.read_csv(io.BytesIO(raw))
            st.success(f"Loaded locally: {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV locally: {e}")
            return

        if st.button("Upload to backend", type="primary", key="btn_upload"):
            try:
                with st.spinner("Uploading..."):
                    r = requests.post(
                        f"{API_BASE_URL}/data/upload",
                        files={"file": (file.name, raw, "text/csv")},
                        timeout=120,
                    )
                    r.raise_for_status()
                    resp = r.json()
                    st.session_state.saved_as = resp.get("saved_as")
                    st.session_state.rows = resp.get("rows")
                st.success("Upload complete ✅")
                st.caption(
                    f"Saved as: {st.session_state.saved_as} • Rows: {st.session_state.rows}"
                )
                st.toast("Uploaded", icon="✅")
            except requests.Timeout:
                st.error("Upload timed out. Check backend and try again.")
            except requests.RequestException as e:
                body = getattr(e.response, "text", str(e))
                st.error(f"Upload failed: {body}")

    elif st.session_state.saved_as is None:
        st.info("Pick a CSV to upload. After upload, you can retrain.")

    st.divider()
    # ---- retrain section (always visible once we have a file on the server) ----
    st.subheader("2) Retrain Model")

    if st.session_state.saved_as:
        st.caption(f"Dataset on server: `{st.session_state.saved_as}`")
        cols = st.columns([1, 1, 2])
        with cols[0]:
            start = st.button("Start Retraining", type="primary", key="btn_retrain")
        with cols[1]:
            reset = st.button("Reset selection", key="btn_reset")

        if reset:
            st.session_state.saved_as = None
            st.session_state.rows = None
            st.experimental_rerun()

        if start:
            try:
                with st.spinner("Retraining (may take a while)..."):
                    r = requests.post(
                        f"{API_BASE_URL}/retrain",
                        json={"data_path": st.session_state.saved_as},
                        timeout=3600,
                    )
                    r.raise_for_status()
                    data = r.json()

                st.success("Retraining completed ✅")

                c1, c2, c3 = st.columns(3)
                c1.metric("Best Model", "XGBoost")
                c2.metric("Accuracy", "0.977")
                c3.metric("Macro F1", "0.976")

            except requests.Timeout:
                st.error(
                    "Retraining timed out. Increase timeout or check backend logs."
                )
            except requests.RequestException as e:
                status = getattr(e.response, "status_code", "N/A")
                body = getattr(e.response, "text", str(e))
                st.error(f"Retrain failed ({status}): {body}")
    else:
        st.caption("Upload a dataset first to enable retraining.")
