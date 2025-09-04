import base64
import os
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")


@st.cache_data(ttl=30)
def _fetch_metrics():
    url = f"{API_BASE_URL}/metrics"
    resp = requests.get(f"{API_BASE_URL}/metrics?embed_images=true", timeout=10)

    try:
        payload = resp.json()
    except Exception:
        payload = {"detail": f"Non-JSON response from {url}"}
    return resp.status_code, payload


def show():
    st.header("📊 Model Performance Metrics")
    code, data = _fetch_metrics()

    if code != 200:
        msg = data.get("detail") if isinstance(data, dict) else str(data)
        st.warning(f"Metrics not available ({code}). {msg}")
        st.info("Tip: re-run your training script to generate api/metrics/metrics.json")
        return

    # Top cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", data.get("best_model", "N/A"))
    with col2:
        st.metric("Accuracy", f"{data.get('accuracy', 0):.3f}")
    with col3:
        st.metric("Macro F1", f"{data.get('macro_f1', 0):.3f}")

    # Per-class table
    st.subheader("Per-Class Metrics")
    per_class = data.get("per_class", [])
    if per_class:
        df = pd.DataFrame(per_class)[["label", "precision", "recall", "f1", "support"]]
        # Format numbers to 3 decimals (leave support as int)
        fmt = {k: "{:.3f}".format for k in ["precision", "recall", "f1"]}
        df_display = df.copy()
        for k, f in fmt.items():
            df_display[k] = df_display[k].map(lambda x: f(x))
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No per-class metrics found in metrics.json.")

    # Raw JSON (optional)
    with st.expander("Raw metrics.json"):
        st.code(json.dumps(data, indent=2), language="json")

    # Download button (optional)
    st.download_button(
        label="Download metrics.json",
        data=json.dumps(data, indent=2),
        file_name="metrics.json",
        mime="application/json",
        use_container_width=True,
    )

    st.subheader("Confusion Matrix")
    cm = data.get("confusion_matrix", {})
    if "png_b64" in cm:
        st.image(
            base64.b64decode(cm["png_b64"]),
            caption="Confusion Matrix",
            use_container_width=True,
        )
    elif "matrix" in cm:
        st.write("Image not embedded; showing numbers instead:")
        st.dataframe(pd.DataFrame(cm["matrix"]), use_container_width=True)
    else:
        st.info("Confusion matrix not found. Re-run training.")
