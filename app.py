import streamlit as st
import numpy as np
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from alert_engine import generate_alert

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Network IDS Dashboard",
    layout="wide",
    page_icon="🛡️"
)

# -------------------------------
# UI STYLE
# -------------------------------
st.markdown("""
<style>
    .main {background-color: #0e1117;}

    h1 {
        text-align: center;
        color: #4fc3f7;
    }

    .stButton>button {
        background-color: #4fc3f7;
        color: black;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Network Intrusion Detection System (CNN-LSTM)")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model("model/model.h5")

# -------------------------------
# LOAD DATA (REAL FIX)
# -------------------------------
X_test = np.load("model/X_test.npy") if os.path.exists("model/X_test.npy") else None
y_test = np.load("model/y_test.npy") if os.path.exists("model/y_test.npy") else None
y_pred = np.load("model/y_pred.npy") if os.path.exists("model/y_pred.npy") else None

# -------------------------------
# LOAD HISTORY
# -------------------------------
history = None
if os.path.exists("model/history.json"):
    with open("model/history.json", "r") as f:
        history = json.load(f)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("System Info")
st.sidebar.write("CNN-LSTM IDS System")
st.sidebar.write("Dataset: CICIDS2017")
st.sidebar.write("Mode: Real Data Inference")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Prediction",
    "📊 Confusion Matrix",
    "📈 Training Performance"
])

# =====================================================
# TAB 1 - REAL PREDICTION (FIXED)
# =====================================================
with tab1:
    st.subheader("Live Network Traffic Analysis")

    if X_test is None:
        st.error("X_test.npy not found. Please retrain model.")
    else:

        idx = st.slider("Select Traffic Sample", 0, len(X_test)-1, 0)

        sample = X_test[idx:idx+1]

        pred = model.predict(sample, verbose=0)

        confidence = float(np.max(pred))
        result = int(np.argmax(pred))

        # ALERT ENGINE
        alert = generate_alert(result, confidence)

        st.markdown("### 🚨 Security Alert Output")

        # SOC STYLE CARD (NO JSON)
        st.markdown(f"""
        <div style="
            padding:15px;
            border-radius:10px;
            background-color:#1e1e2f;
            border:1px solid #4fc3f7;
            color:white;
        ">
        <b>Status:</b> {alert['status']}<br>
        <b>Type:</b> {alert['type']}<br>
        <b>Severity:</b> {alert['severity']}<br>
        <b>Confidence:</b> {alert['confidence']}<br>
        <b>Action:</b> {alert['action']}<br>
        </div>
        """, unsafe_allow_html=True)

        if alert["status"] == "ATTACK DETECTED":
            st.error("⚠ Threat Detected in Network Traffic")
        else:
            st.success("✔ Normal Network Behavior")

# =====================================================
# TAB 2 - CONFUSION MATRIX
# =====================================================
with tab2:
    st.subheader("Model Evaluation")

    if y_test is not None and y_pred is not None:

        cm = confusion_matrix(y_test, y_pred)

        # SMALL + CONTROLLED FIGURE
        fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=140)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["BENIGN", "ATTACK"],
            yticklabels=["BENIGN", "ATTACK"],
            linewidths=0.5,
            cbar=False,
            ax=ax
        )

        ax.set_title("Confusion Matrix", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

        ax.tick_params(axis='both', labelsize=8)

        # FORCE SMALL DISPLAY
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig)

    else:
        st.warning("No test data available.")

# =====================================================
# TAB 3 - TRAINING CURVE
# =====================================================
with tab3:
    st.subheader("Model Learning Curve")

    if history is not None:

        acc = history.get("accuracy", [])
        val_acc = history.get("val_accuracy", [])

        if len(acc) > 0:

            fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=140)

            ax.plot(acc, label="Train Accuracy", linewidth=2)
            ax.plot(val_acc, label="Validation Accuracy", linewidth=2)

            ax.set_title("Learning Curve", fontsize=9)
            ax.set_xlabel("Epochs", fontsize=8)
            ax.set_ylabel("Accuracy", fontsize=8)

            ax.tick_params(axis='both', labelsize=8)

            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # CENTER IT LIKE DASHBOARD WIDGET
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig)

            st.success(f"Final Accuracy: {acc[-1]*100:.2f}%")

        else:
            st.error("Empty training history.")

    else:
        st.warning("No training history found.")