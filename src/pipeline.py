# src/pipeline.py

from typing import Dict, Any, List

from src.model_service import DiseasePredictor
from src.nhanes_engine import LifestyleRecommender, UserProfile
from src.llm_service import generate_recommendation_with_gemini

predictor = DiseasePredictor()
recommender = LifestyleRecommender()


def chat_pipeline(
    user_text: str,
    age: int,
    sex: str,
    sugar: float | None,
    sodium: float | None,
    bmi: float | None,
    activity: float | None,
    smoking: str | None,
    fiber: float | None,
    protein: float | None,
    alcohol: float | None,
    extra_structured: Dict[str, Any] | None = None,
) -> str:
    """
    1. Use Random Forest to predict top disease risks.
    2. Build lifestyle context from age + metrics (sugar, sodium, activity, etc.).
    3. Ask Gemini to:
       - Summarize RF risk table (not a diagnosis).
       - Provide tailored lifestyle recommendations based on thresholds/targets.
    """

    extra_structured = extra_structured or {}

    # -----------------------------
    # 1️⃣ Build categorical inputs for RF
    # -----------------------------
    cats = {
        "gender": sex if sex and sex != "Unknown" else None
    }

    # -----------------------------
    # 2️⃣ Build numeric inputs dynamically using predictor.num_cols
    # -----------------------------
    nums: Dict[str, Any] = {}

    for col in predictor.num_cols:
        if col == "age":
            nums[col] = age

        elif col == "bmi":
            nums[col] = bmi

        elif col == "weekly_activity_mins":
            nums[col] = activity

        # diet-related features from user if they are part of num_cols
        elif col == "sugar_g":
            nums[col] = sugar
        elif col == "sodium_mg":
            nums[col] = sodium
        elif col == "fiber_g":
            nums[col] = fiber
        elif col == "protein_g":
            nums[col] = protein
        elif col == "alcohol_g":
            nums[col] = alcohol

        # any lab features (e.g. mean_glucose, mean_creatinine, etc.)
        elif col in extra_structured:
            nums[col] = extra_structured[col]

        else:
            # Let the ColumnTransformer impute missing numeric features
            nums[col] = None

    # -----------------------------
    # 3️⃣ Disease prediction via RF
    # -----------------------------
    preds = predictor.predict_from_inputs(
        text=user_text,
        cats=cats,
        nums=nums,
        k=3,
        threshold=0.05,   # lower threshold so you usually get a few candidates
    )

    if not preds:
        return (
            "I couldn't confidently infer any specific disease-related risk from your description. "
            "Please share more detail about your symptoms, age, and any known lab values, and be sure to consult a clinician."
        )

    # preds: List[Tuple[disease_name, prob]]
    risk_table: List[Dict[str, Any]] = [
        {"disease": d, "prob": float(p)} for d, p in preds
    ]
    top_disease, top_prob = preds[0]

    # -----------------------------
    # 4️⃣ Build NHANES lifestyle profile (disease-agnostic)
    # -----------------------------
    profile = UserProfile(
        age=int(age),
        sex=None if sex == "Unknown" else sex,
        sugar_g_day=sugar,
        sodium_mg_day=sodium,
        bmi=bmi,
        activity_minutes_week=activity,
        smoking_status=None if smoking == "Unknown" else smoking,
        fiber_g_day=fiber,
        protein_g_day=protein,
        alcohol_g_day=alcohol,
    )

    nhanes_ctx = recommender.build_context(top_disease, profile)

    # -----------------------------
    # 5️⃣ Generate RF risk explanation + lifestyle recommendations via Gemini
    # -----------------------------
    reply = generate_recommendation_with_gemini(
        user_query=user_text,
        risk_table=risk_table,
        nhanes_context=nhanes_ctx,
    )

    return reply
