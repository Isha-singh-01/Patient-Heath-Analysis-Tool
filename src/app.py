# app.py

import os
import sys
from pathlib import Path

import streamlit as st

# -------------------------------------------------------------------
# import from src/
# -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.pipeline import chat_pipeline  # uses RF + NHANES + Gemini


# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Health Lifestyle Assistant (Prototype)",
    page_icon="🩺",
    layout="wide",
)

# -------------------------------------------------------------------
# Sidebar: user profile & lifestyle inputs
# -------------------------------------------------------------------
st.sidebar.title("🧍‍♂️ Patient Profile")

st.sidebar.markdown(
    "Provide a **rough profile** so the assistant can "
    "tailor its lifestyle recommendations. Values don’t have to be perfect."
)

age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1)

sex = st.sidebar.selectbox(
    "Sex",
    options=["M", "F", "Unknown"],
    index=0,
    help="Use the biological sex variable used in your model.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🍽️ Diet (per day)")

sugar = st.sidebar.number_input(
    "Added sugar (g/day)",
    min_value=0.0,
    max_value=500.0,
    value=120.0,
    step=1.0,
    help="Approximate total added sugars from drinks, desserts, etc.",
)

sodium = st.sidebar.number_input(
    "Sodium (mg/day)",
    min_value=0.0,
    max_value=10000.0,
    value=2300.0,
    step=50.0,
    help="Total daily sodium, including processed foods.",
)

fiber = st.sidebar.number_input(
    "Fiber (g/day)",
    min_value=0.0,
    max_value=100.0,
    value=12.0,
    step=0.5,
    help="Dietary fiber from fruits, vegetables, whole grains, etc.",
)

protein = st.sidebar.number_input(
    "Protein (g/day)",
    min_value=0.0,
    max_value=300.0,
    value=60.0,
    step=1.0,
    help="Approximate total protein intake per day.",
)

alcohol = st.sidebar.number_input(
    "Alcohol (g/day)",
    min_value=0.0,
    max_value=200.0,
    value=16.0,
    step=1.0,
    help="0 if you do not drink; otherwise approximate daily average.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🏃 Activity & Body Composition")

activity = st.sidebar.number_input(
    "Physical activity (minutes/week)",
    min_value=0.0,
    max_value=2000.0,
    value=20.0,
    step=5.0,
    help="Total moderate/vigorous activity per week.",
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=28.5,
    step=0.1,
    help="If unknown, you can approximate using online BMI calculators.",
)

smoking = st.sidebar.selectbox(
    "Smoking status",
    options=["none", "current", "former", "Unknown"],
    index=0,
    help="Used only for lifestyle recommendations, not diagnosis.",
)

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ This is a research prototype. It does **not** provide medical "
    "diagnoses or treatment. Always consult a clinician for health concerns."
)

# -------------------------------------------------------------------
# Session state for chat history
# -------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm a **lifestyle-focused health assistant prototype**.\n\n"
                "👉 I can:\n"
                "- Estimate **relative disease risks** using a trained model (not a diagnosis).\n"
                "- Use your **diet, activity, and other factors** to suggest small, realistic lifestyle steps.\n\n"
                "Please describe your symptoms or concerns, and make sure your profile in the sidebar looks roughly correct."
            ),
        }
    ]


# -------------------------------------------------------------------
# Main layout
# -------------------------------------------------------------------
st.title("🩺 Health Risk & Lifestyle Recommendation Chatbot")

st.caption(
    "Bridging disease risk prediction (MIMIC-IV Random Forest) with "
    "NHANES-based lifestyle guidance, powered by Gemini."
)

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        """
        - A multiclass **Random Forest** model (trained on MIMIC-IV) estimates disease **risk scores**
          from your description + basic features (age, sex, etc.).  
        - A separate **lifestyle engine** uses **NHANES 2021–2023** distributions and guideline thresholds  
          (for sugar, sodium, activity, BMI, fiber, protein, alcohol, etc.)  
          to understand where you are vs. where you *could* be.  
        - A **Gemini 2.5 Pro** LLM turns those numbers into a readable explanation with **2–5 realistic steps**.  

        This is for educational / research purposes only and is **not medical advice**.
        """
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
user_input = st.chat_input("Describe your symptoms, concerns, or goals (e.g., chest pain, fatigue, wanting to improve heart health)...")

if user_input:
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Call backend pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking about your risk profile and lifestyle..."):
            try:
                reply = chat_pipeline(
                    user_text=user_input,
                    age=int(age),
                    sex=sex,
                    sugar=float(sugar),
                    sodium=float(sodium),
                    bmi=float(bmi),
                    activity=float(activity),
                    smoking=smoking,
                    fiber=float(fiber),
                    protein=float(protein),
                    alcohol=float(alcohol),
                )
            except Exception as e:
                reply = (
                    "I ran into an internal error while processing your request. "
                    "Please try again, and if it keeps happening, let the developers know.\n\n"
                    f"Error details: `{e}`"
                )

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
