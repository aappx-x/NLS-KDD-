"""
Streamlit demo for the ML-based Network Intrusion Detection prototype.

What this app does:
  1. Lets the user pick a preset example (one per NSL-KDD attack category) or
     manually edit the 40 network-flow features.
  2. Runs the SAME preprocessing pipeline (preprocessor.joblib) that was
     fitted during training — this is important: the app must transform new
     input exactly the way training data was transformed, or predictions
     would be meaningless.
  3. Runs the trained Random Forest and shows the prediction, the model's
     confidence, and the top features driving importance in general.

What this app is NOT:
  - It does not capture live network traffic.
  - It does not act on predictions (no blocking, no alerting pipeline).
  - It is a demonstration of the trained model on NSL-KDD-style feature
    vectors, not a production intrusion detection system.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"

st.set_page_config(page_title="NIDS Prototype", layout="wide")


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    raw_feature_cols = joblib.load(MODELS_DIR / "raw_feature_cols.joblib")

    tuned_path = MODELS_DIR / "random_forest_tuned.joblib"
    model_path = tuned_path if tuned_path.exists() else MODELS_DIR / "random_forest_untuned.joblib"
    model = joblib.load(model_path)

    importance_path = MODELS_DIR / "importance_permutation.joblib"
    importance = joblib.load(importance_path) if importance_path.exists() else None

    return preprocessor, raw_feature_cols, model, importance, model_path.name


@st.cache_data
def load_presets():
    with open(APP_DIR / "sample_presets.json") as f:
        presets = json.load(f)
    with open(APP_DIR / "categorical_options.json") as f:
        cat_options = json.load(f)
    return presets, cat_options


preprocessor, raw_feature_cols, model, importance, model_filename = load_artifacts()
presets, cat_options = load_presets()

st.title("Network Intrusion Detection — ML Prototype")
st.caption(
    "This is a machine-learning demonstration built on the offline NSL-KDD dataset. "
    "It is **not** a production-grade or real-time IDS — see the Limitations section below."
)

with st.sidebar:
    st.header("Input")
    preset_name = st.selectbox(
        "Load an example connection record",
        options=["-- manual entry --"] + list(presets.keys()),
        help="Each preset is a real row from the NSL-KDD test set for that attack category.",
    )
    st.caption(f"Model in use: `{model_filename}`")

if preset_name != "-- manual entry --":
    base_values = presets[preset_name]
else:
    base_values = presets["normal"]  # sensible starting point for manual editing

st.subheader("Network-flow features")
col1, col2, col3 = st.columns(3)

user_input = {}

with col1:
    st.markdown("**Connection basics**")
    user_input["duration"] = st.number_input("duration (seconds)", min_value=0, value=int(base_values["duration"]))
    user_input["protocol_type"] = st.selectbox(
        "protocol_type", cat_options["protocol_type"],
        index=cat_options["protocol_type"].index(base_values["protocol_type"]),
    )
    user_input["service"] = st.selectbox(
        "service", cat_options["service"],
        index=cat_options["service"].index(base_values["service"]),
    )
    user_input["flag"] = st.selectbox(
        "flag", cat_options["flag"],
        index=cat_options["flag"].index(base_values["flag"]),
    )
    user_input["src_bytes"] = st.number_input("src_bytes", min_value=0, value=int(base_values["src_bytes"]))
    user_input["dst_bytes"] = st.number_input("dst_bytes", min_value=0, value=int(base_values["dst_bytes"]))

with col2:
    st.markdown("**Traffic-rate features (last 2s window)**")
    user_input["count"] = st.number_input("count (connections to same host)", min_value=0, value=int(base_values["count"]))
    user_input["srv_count"] = st.number_input("srv_count (connections to same service)", min_value=0, value=int(base_values["srv_count"]))
    user_input["serror_rate"] = st.slider("serror_rate", 0.0, 1.0, float(base_values["serror_rate"]))
    user_input["rerror_rate"] = st.slider("rerror_rate", 0.0, 1.0, float(base_values["rerror_rate"]))
    user_input["same_srv_rate"] = st.slider("same_srv_rate", 0.0, 1.0, float(base_values["same_srv_rate"]))
    user_input["diff_srv_rate"] = st.slider("diff_srv_rate", 0.0, 1.0, float(base_values["diff_srv_rate"]))

with col3:
    st.markdown("**Host-based & content features**")
    user_input["logged_in"] = st.selectbox("logged_in", [0, 1], index=int(base_values["logged_in"]))
    user_input["num_failed_logins"] = st.number_input("num_failed_logins", min_value=0, value=int(base_values["num_failed_logins"]))
    user_input["hot"] = st.number_input("hot (suspicious ops count)", min_value=0, value=int(base_values["hot"]))
    user_input["dst_host_count"] = st.number_input("dst_host_count", min_value=0, max_value=255, value=int(base_values["dst_host_count"]))
    user_input["dst_host_srv_count"] = st.number_input("dst_host_srv_count", min_value=0, max_value=255, value=int(base_values["dst_host_srv_count"]))
    user_input["dst_host_serror_rate"] = st.slider("dst_host_serror_rate", 0.0, 1.0, float(base_values["dst_host_serror_rate"]))

with st.expander("Show / edit all 40 raw features (advanced)"):
    full_row = dict(base_values)
    full_row.update(user_input)
    edited_df = st.data_editor(
        pd.DataFrame([full_row])[raw_feature_cols], num_rows="fixed", key="full_editor"
    )
    full_row = edited_df.iloc[0].to_dict()

# Merge: values from the quick-edit columns take priority, remaining raw
# features come from the preset / advanced editor.
final_row = dict(full_row)
final_row.update(user_input)

st.divider()

if st.button("Run prediction", type="primary"):
    input_df = pd.DataFrame([final_row])[raw_feature_cols]

    X = preprocessor.transform(input_df)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    result_col, importance_col = st.columns([1, 1])

    with result_col:
        if pred == 1:
            st.error(f"### Prediction: ATTACK\nConfidence: {proba[1]*100:.1f}%")
        else:
            st.success(f"### Prediction: NORMAL\nConfidence: {proba[0]*100:.1f}%")

        st.write("Class probabilities:")
        st.bar_chart(pd.DataFrame({"probability": [proba[0], proba[1]]}, index=["normal", "attack"]))

    with importance_col:
        st.write("**Top features the model relies on in general**")
        st.caption("(Permutation importance computed once on the test set — not specific to this single input.)")
        if importance is not None:
            st.bar_chart(importance.head(10))
        else:
            st.info("Run src/feature_importance.py to generate this chart.")

st.divider()
with st.expander("About this demo / limitations"):
    st.markdown(
        """
- Trained and evaluated on **NSL-KDD**, an offline, publicly available, labeled
  dataset from simulated military network traffic — not live production traffic.
- The model outputs a **binary** prediction (normal vs. attack) from a single
  aggregated connection record; it does not classify attack *type* in this demo,
  though the underlying dataset supports that as a future extension.
- Performance figures (precision/recall/F1/ROC-AUC) are reported in the README
  and were measured on NSL-KDD's held-out test set, not on real network traffic.
- This prototype does **not** claim real-time capability, production readiness,
  or the ability to reliably detect previously unseen (zero-day) attack types.
        """
    )
