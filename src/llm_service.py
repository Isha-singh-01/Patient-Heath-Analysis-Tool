# src/llm_service.py
import os
import json
import textwrap
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-pro"

def generate_recommendation_with_gemini(user_query: str,
                                        disease: str,
                                        prob: float,
                                        nhanes_context: dict) -> str:
    """
    Calls Gemini to turn structured NHANES context into a human-friendly explanation.
    """
    nhanes_json = json.dumps(nhanes_context, indent=2)

    system_prompt = textwrap.dedent("""
        You are a health lifestyle assistant. 
        You DO NOT diagnose or prescribe medications.
        You provide lifestyle recommendations only (diet, activity, smoking cessation, sleep, etc.).
        You will be given:
        - The user's query.
        - A predicted disease risk category with confidence.
        - A JSON object containing NHANES-based statistics, user behavior estimates, 
          guideline thresholds, and intermediate targets.

        Rules:
        - Use only the numeric values from the JSON object when mentioning numbers.
        - Explain the situation in clear, non-technical language.
        - Focus on realistic, incremental changes (e.g., reducing sugar gradually).
        - Provide 2–4 specific suggestions, referencing the intermediate targets where given.
        - Always remind the user that this is not medical advice and they should consult a clinician.
        - If a metric is missing, you can ask the user to provide that information instead of guessing.
    """)

    user_prompt = f"""
    User's original query:
    {user_query}

    Predicted disease category (not a diagnosis):
    {disease} (model confidence: {prob:.2f})

    NHANES and guideline context (JSON):
    {nhanes_json}

    Please:
    1. Briefly describe what the numbers indicate about their current risk-related habits.
    2. Provide 2–4 concrete, achievable lifestyle recommendations (diet, sodium, physical activity, etc.).
    3. Use a reassuring, non-alarming tone.
    4. End with a one-line reminder that this is not medical advice.
    """

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        [system_prompt, user_prompt],
        safety_settings=None,  # or adjust if you want
    )

    return response.text.strip()