import streamlit as st
import numpy as np
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="IoT Botnet Detection System",
    layout="wide",
    page_icon="🛡️"
)

# -------------------------------
# PROFESSIONAL UI STYLE (IMPROVED)
# -------------------------------
st.markdown("""
<style>
    .main {background-color: #0e1117;}

    h1 {
        text-align: center;
        color: #4fc3f7;
    }

    h2, h3 {
        color: #4fc3f7;
    }

    .stButton>button {
        background-color: #4fc3f7;
        color: black;
        border-radius: 12px;
        height: 3.2em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }

    .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ IoT Botnet Detection Dashboard")
st.markdown("### Hybrid CNN-based Intrusion Detection (CICIDS2017)")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model("model/model.h5")

# -------------------------------
# LOAD HISTORY
# -------------------------------
history_path = "model/history.json"
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)
else:
    history = None

# -------------------------------
# LOAD DATA
# -------------------------------
y_test = np.load("model/y_test.npy") if os.path.exists("model/y_test.npy") else None
y_pred = np.load("model/y_pred.npy") if os.path.exists("model/y_pred.npy") else None

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("System Info")
st.sidebar.write("Model: CNN-based IDS")
st.sidebar.write("Dataset: CICIDS2017")
st.sidebar.write("Classes: BENIGN / ATTACK")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Prediction",
    "📊 Confusion Matrix",
    "📈 Training Performance"
])

# =====================================================
# TAB 1 - PREDICTION (FIXED LAYOUT)
# =====================================================
with tab1:
    st.subheader("Traffic Classification Demo")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### Normal Traffic")

        if st.button("Test Normal Traffic"):
            sample = np.zeros((1, 79, 1))
            pred = model.predict(sample, verbose=0)
            result = np.argmax(pred)

            st.metric("Confidence", f"{np.max(pred)*100:.2f}%")

            if result == 0:
                st.success("🟢 BENIGN TRAFFIC")
            else:
                st.error("🔴 ATTACK DETECTED")

    with col2:
        st.markdown("### Attack Traffic")

        if st.button("Test Attack Traffic"):
            sample = np.ones((1, 79, 1))
            pred = model.predict(sample, verbose=0)
            result = np.argmax(pred)

            st.metric("Confidence", f"{np.max(pred)*100:.2f}%")

            if result == 0:
                st.success("🟢 BENIGN TRAFFIC")
            else:
                st.error("🔴 ATTACK DETECTED")

# =====================================================
# TAB 2 - CONFUSION MATRIX (FIXED SIZE)
# =====================================================
with tab2:
    st.subheader("Confusion Matrix (Model Evaluation)")

    if y_test is not None and y_pred is not None:

        cm = confusion_matrix(y_test, y_pred)

        # 🔥 SMALL FIXED CANVAS
        fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=120)

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

        ax.set_title("Confusion Matrix", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

        ax.tick_params(axis='both', labelsize=9)

        # 🔥 CRITICAL: DO NOT let Streamlit resize it
        st.pyplot(fig, use_container_width=False)

# =====================================================
# TAB 3 - TRAINING GRAPH (FIXED VISUAL SCALE)
# =====================================================
with tab3:
    st.subheader("Model Learning Curve")

    if history is not None:

        acc = history.get("accuracy", [])
        val_acc = history.get("val_accuracy", [])

        if len(acc) == 0:
            st.error("No training data found.")
        else:

            # 🔥 SMALL CONTROLLED FIGURE
            fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)

            ax.plot(acc, label="Training Accuracy", linewidth=2)
            ax.plot(val_acc, label="Validation Accuracy", linewidth=2)

            ax.set_xlabel("Epochs", fontsize=9)
            ax.set_ylabel("Accuracy", fontsize=9)
            ax.set_title("Model Learning Curve", fontsize=10)

            ax.tick_params(axis='both', labelsize=8)

            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # 🔥 IMPORTANT: stop Streamlit from stretching it
            st.pyplot(fig, use_container_width=False)

            st.success(f"Final Accuracy: {acc[-1]*100:.2f}%")

    else:
        st.warning("No training history found.")