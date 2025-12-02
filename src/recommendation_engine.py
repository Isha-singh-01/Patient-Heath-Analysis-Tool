import pandas as pd
import numpy as np
import json
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
from load_env import load_env, get_config
load_env()  # This loads .env file automatically

# Import disease mapping (if available)
try:
    from disease_mapping import COMPREHENSIVE_DISEASE_MAPPING, DISEASE_CLINICAL_CONTEXT, get_profile_for_disease
    HAS_DISEASE_MAPPING = True
except ImportError:
    HAS_DISEASE_MAPPING = False
    print("⚠️  disease_mapping.py not found - using simplified disease profiles")

# Import enhanced lifestyle engine
from nhanes_lifestyle_engine import EnhancedLifestyleRecommender, UserMetrics

class ComprehensiveRecommendationSystem:
    """
    Complete end-to-end recommendation system integrating:
    - MIMIC-IV trained Random Forest for disease prediction
    - NHANES 2021-2023 for population comparison
    - Hugging Face LLM for personalized recommendations
    """
    
    def __init__(self, 
                 model_dir='models_fixed',
                 profiles_path='models_fixed/comprehensive_disease_profiles.json',
                 use_llm=True,
                 hf_api_key=None,
                 hf_model='meta-llama/Llama-3.2-3B-Instruct'):
        
        print("="*70)
        print("INITIALIZING COMPREHENSIVE RECOMMENDATION SYSTEM")
        print("="*70)
        
        model_dir = Path(model_dir)
        
        # Load Random Forest model
        print("\n1. Loading Random Forest model...")
        model_path = model_dir / 'disease_rf_model.pkl'
        if not model_path.exists():
            model_path = model_dir / 'random_forest_model.pkl'
        
        with open(model_path, 'rb') as f:
            self.rf_model = pickle.load(f)
        print(f"   ✓ Random Forest loaded from {model_path.name}")
        
        # Load TF-IDF vectorizer
        print("\n2. Loading TF-IDF vectorizer...")
        tfidf_path = model_dir / 'tfidf_vectorizer.pkl'
        with open(tfidf_path, 'rb') as f:
            self.tfidf = pickle.load(f)
        print(f"   ✓ TF-IDF loaded (max_features: {self.tfidf.max_features})")
        
        # Load Column Transformer
        print("\n3. Loading Column Transformer...")
        ct_path = model_dir / 'column_transformer.pkl'
        with open(ct_path, 'rb') as f:
            self.column_transformer = pickle.load(f)
        print(f"   ✓ Column Transformer loaded")
        
        # Load Label Encoder
        print("\n4. Loading Label Encoder...")
        le_path = model_dir / 'label_encoder.pkl'
        with open(le_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Get disease names from label encoder
        self.disease_names = list(self.label_encoder.classes_)
        print(f"   ✓ Label Encoder loaded")
        print(f"   ✓ Disease names loaded: {len(self.disease_names)} classes")
        
        # Also load metadata for other info
        disease_names_path = model_dir / 'disease_rf_metadata.json'
        if disease_names_path.exists():
            with open(disease_names_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Load NHANES disease profiles
        print("\n5. Loading NHANES disease profiles...")
        profiles_path = Path(profiles_path)
        if not profiles_path.exists():
            profiles_path = Path('models_fixed/comprehensive_disease_profiles.json')
        
        with open(profiles_path, 'r') as f:
            self.profiles = json.load(f)
        print(f"   ✓ Loaded {len(self.profiles)} disease profiles")
        
        # LLM configuration
        self.use_llm = use_llm
        self.hf_api_key = hf_api_key
        self.hf_model = hf_model
        # Updated endpoint - HF changed their API URL
        self.hf_api_url = f"https://api-inference.huggingface.co/models/{hf_model}"
        
        if self.use_llm:
            print(f"\n6. LLM Configuration:")
            print(f"   Provider: Hugging Face Inference API")
            print(f"   Model: {hf_model}")
            if not hf_api_key:
                print(f"   ⚠️  No API key provided - set HF_API_KEY environment variable")
            else:
                print(f"   ✓ API key configured")
        else:
            print(f"\n6. LLM: Disabled (using rule-based recommendations)")
        
        # Initialize enhanced lifestyle recommender
        self.lifestyle_recommender = EnhancedLifestyleRecommender()
        
        print("\n" + "="*70)
        print("✅ SYSTEM INITIALIZED SUCCESSFULLY")
        print("="*70)
    
    def prepare_input_features(self, user_input: Dict) -> tuple:
        """
        Prepare input features matching your model's training format
        
        Args:
            user_input: {
                'age': int,
                'gender': str ('M' or 'F'),
                'symptoms_text': str,  # Natural language symptoms
                'glucose': float (optional),
                'hematocrit': float (optional),
                'hemoglobin': float (optional),
                'creatinine': float (optional),
                'potassium': float (optional),
                'sodium': float (optional),
                'urea_nitrogen': float (optional)
            }
        
        Returns:
            Combined sparse feature matrix (text + structured)
        """
        # Standardize gender
        gender = user_input.get('gender', 'M').upper()
        if gender not in ['M', 'F']:
            gender = 'M'
        
        age = user_input.get('age', 40)
        
        # Create structured data DataFrame
        structured_data = pd.DataFrame({
            'gender': [gender],
            'age': [age]
        })
        
        # Add lab values (matching your training columns)
        lab_features = {}
        
        # Your model uses mean/min/max for each lab
        # We'll use the user's value for all three (mean=min=max)
        lab_mapping = {
            'glucose': 'Glucose',
            'hematocrit': 'Hematocrit',
            'hemoglobin': 'Hemoglobin',  # Added - model expects this
            'creatinine': 'Creatinine',
            'potassium': 'Potassium',
            'sodium': 'Sodium',
            'urea_nitrogen': 'Urea Nitrogen'
        }
        
        # Default values for missing features (normal ranges)
        defaults = {
            'Glucose': 100.0,
            'Hematocrit': 42.0,
            'Hemoglobin': 14.0,
            'Creatinine': 1.0,
            'Potassium': 4.0,
            'Sodium': 140.0,
            'Urea Nitrogen': 15.0
        }
        
        # Add all required features (provided or default)
        for input_key, lab_name in lab_mapping.items():
            if input_key in user_input and user_input[input_key] is not None:
                value = user_input[input_key]
            else:
                value = defaults.get(lab_name, 0.0)
            
            structured_data[f'mean_{lab_name}'] = value
            structured_data[f'min_{lab_name}'] = value
            structured_data[f'max_{lab_name}'] = value
        
        # Text features
        symptoms_text = user_input.get('symptoms_text', '')
        if not symptoms_text and 'symptoms' in user_input:
            # Convert list to text
            symptoms_text = ' '.join(user_input['symptoms'])
        
        # Apply text cleaning (match your training pipeline)
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        import re
        
        lemmatizer = WordNetLemmatizer()
        stop = set(stopwords.words('english'))
        
        def clean_text(s: str) -> str:
            s = str(s).lower()
            s = re.sub(r'[^a-z\s]', ' ', s)
            tokens = [w for w in s.split() if w not in stop and len(w) > 2]
            lemmas = [lemmatizer.lemmatize(w) for w in tokens]
            return " ".join(lemmas)
        
        text_clean = clean_text(symptoms_text)
        
        # Transform text with TF-IDF
        text_features = self.tfidf.transform([text_clean])
        
        # Transform structured data
        structured_features = self.column_transformer.transform(structured_data)
        
        # Combine
        combined_features = hstack([text_features, structured_features])
        
        return combined_features
    
    def predict_disease(self, user_input: Dict) -> Tuple[str, float, np.ndarray, Dict]:
        """
        Step 1: Predict disease using Random Forest model
        
        Returns:
            (predicted_disease, confidence, all_probabilities, top_3_predictions)
        """
        print("\n" + "-"*70)
        print("STEP 1: DISEASE PREDICTION (Random Forest)")
        print("-"*70)
        
        # Prepare features
        features = self.prepare_input_features(user_input)
        
        # Predict
        probabilities = self.rf_model.predict_proba(features)[0]
        predicted_class_idx = self.rf_model.predict(features)[0]
        predicted_disease = self.disease_names[predicted_class_idx]
        confidence = probabilities.max()
        
        print(f"✓ Predicted Disease: {predicted_disease}")
        print(f"✓ Confidence: {confidence*100:.1f}%")
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3 = {}
        
        print(f"\nTop 3 Predictions:")
        for i, idx in enumerate(top_3_idx, 1):
            disease = self.disease_names[idx]
            prob = probabilities[idx]
            top_3[disease] = float(prob)
            print(f"  {i}. {disease}: {prob*100:.1f}%")
        
        return predicted_disease, confidence, probabilities, top_3
    
    def get_nhanes_profile(self, predicted_disease: str) -> Dict:
        """Step 2: Retrieve NHANES population profile"""
        print("\n" + "-"*70)
        print("STEP 2: RETRIEVING NHANES POPULATION DATA")
        print("-"*70)
        
        print(f"Disease: {predicted_disease}")
        
        # Try to get profile mapping if available
        if HAS_DISEASE_MAPPING:
            try:
                profile_key, clinical_context = get_profile_for_disease(predicted_disease)
                print(f"NHANES Profile: {profile_key}")
                profile = self.profiles.get(profile_key, self.profiles.get('healthy_reference', {}))
                
                if clinical_context:
                    profile['clinical_context'] = clinical_context
                    print(f"Clinical Category: {clinical_context['category']}")
                    print(f"Key Concerns: {', '.join(clinical_context['key_concerns'][:3])}")
            except Exception as e:
                print(f"⚠️  Could not get disease mapping: {e}")
                profile = self.profiles.get(predicted_disease, self.profiles.get('healthy_reference', {}))
        else:
            # Fallback: try direct lookup
            profile = self.profiles.get(predicted_disease, self.profiles.get('healthy_reference', {}))
        
        print(f"✓ Profile loaded: {profile.get('sample_size', 0):,} NHANES participants")
        
        return profile
    
    def compare_user_to_population(self, user_input: Dict, profile: Dict) -> Dict:
        """Step 3: Compare user's values to NHANES population"""
        print("\n" + "-"*70)
        print("STEP 3: POPULATION COMPARISON")
        print("-"*70)
        
        comparisons = {}
        
        # Glucose comparison - ALWAYS include if provided
        if 'glucose' in user_input and user_input['glucose'] is not None:
            glucose_data = profile.get('clinical_markers', {}).get('glucose', {})
            user_glucose = user_input['glucose']
            pop_mean = glucose_data.get('mean', 100) if glucose_data else 100
            target = glucose_data.get('target', '<100 mg/dL') if glucose_data else '<100 mg/dL'
            
            status = 'normal'
            if user_glucose >= 126:
                status = 'diabetic range'
            elif user_glucose >= 100:
                status = 'prediabetic range'
            
            comparisons['glucose'] = {
                'user_value': user_glucose,
                'population_mean': round(pop_mean, 1),
                'target': target,
                'status': status,
                'deviation': round(user_glucose - pop_mean, 1)
            }
            print(f"✓ Glucose: {user_glucose} mg/dL (target: {target}, status: {status})")
        
        # BMI comparison - ALWAYS include if provided
        if 'bmi' in user_input and user_input['bmi'] is not None:
            bmi_data = profile.get('clinical_markers', {}).get('bmi', {})
            user_bmi = user_input['bmi']
            pop_mean = bmi_data.get('mean', 27) if bmi_data else 27
            
            status = 'normal'
            if user_bmi >= 30:
                status = 'obese'
            elif user_bmi >= 25:
                status = 'overweight'
            elif user_bmi < 18.5:
                status = 'underweight'
            
            comparisons['bmi'] = {
                'user_value': user_bmi,
                'population_mean': round(pop_mean, 1),
                'target': '18.5-24.9 kg/m²',
                'status': status
            }
            print(f"✓ BMI: {user_bmi} kg/m² (status: {status})")
        
        # Blood pressure - ALWAYS include if provided
        if 'systolic_bp' in user_input and user_input['systolic_bp'] is not None:
            bp_data = profile.get('clinical_markers', {}).get('blood_pressure', {})
            user_systolic = user_input['systolic_bp']
            pop_mean = bp_data.get('systolic_mean', 120) if bp_data else 120
            
            status = 'normal'
            if user_systolic >= 140:
                status = 'stage 2 hypertension'
            elif user_systolic >= 130:
                status = 'stage 1 hypertension'
            elif user_systolic >= 120:
                status = 'elevated'
            
            comparisons['blood_pressure'] = {
                'user_systolic': user_systolic,
                'user_diastolic': user_input.get('diastolic_bp', 80),
                'population_mean': round(pop_mean, 1),
                'target': '<120/80 mmHg',
                'status': status
            }
            print(f"✓ BP: {user_systolic}/{user_input.get('diastolic_bp', 80)} mmHg (status: {status})")
        
        # Creatinine - ALWAYS include if provided
        if 'creatinine' in user_input and user_input['creatinine'] is not None:
            creat_data = profile.get('clinical_markers', {}).get('creatinine', {})
            user_creat = user_input['creatinine']
            pop_mean = creat_data.get('mean', 0.9) if creat_data else 0.9
            
            status = 'normal'
            if user_creat > 1.5:
                status = 'significantly elevated'
            elif user_creat > 1.2:
                status = 'elevated'
            
            comparisons['creatinine'] = {
                'user_value': user_creat,
                'population_mean': round(pop_mean, 2),
                'target': '<1.2 mg/dL',
                'status': status
            }
            print(f"✓ Creatinine: {user_creat} mg/dL (status: {status})")
        
        # Sodium (serum) - if provided
        if 'sodium' in user_input and user_input['sodium'] is not None and user_input['sodium'] > 0:
            comparisons['sodium_serum'] = {
                'user_value': user_input['sodium'],
                'target': '135-145 mEq/L',
                'status': 'normal' if 135 <= user_input['sodium'] <= 145 else 'abnormal'
            }
            print(f"✓ Sodium (serum): {user_input['sodium']} mEq/L")
        
        # Potassium - if provided
        if 'potassium' in user_input and user_input['potassium'] is not None and user_input['potassium'] > 0:
            comparisons['potassium'] = {
                'user_value': user_input['potassium'],
                'target': '3.5-5.0 mEq/L',
                'status': 'normal' if 3.5 <= user_input['potassium'] <= 5.0 else 'abnormal'
            }
            print(f"✓ Potassium: {user_input['potassium']} mEq/L")
        
        if not comparisons:
            print("⚠️  No clinical markers provided for comparison")
        
        return comparisons
    
    def generate_llm_recommendations(self, 
                                    predicted_disease: str,
                                    confidence: float,
                                    user_input: Dict,
                                    profile: Dict,
                                    comparisons: Dict,
                                    top_3: Dict) -> str:
        """Step 4: Generate personalized recommendations with enhanced lifestyle context"""
        print("\n" + "-"*70)
        print("STEP 4: GENERATING PERSONALIZED RECOMMENDATIONS")
        print("-"*70)
        
        if not self.use_llm:
            print("LLM disabled - using rule-based recommendations")
            return self._generate_fallback_recommendations(predicted_disease, profile, comparisons, top_3, lifestyle_context, user_input)
        
        # Build enhanced lifestyle context
        user_metrics = UserMetrics(
            age=user_input.get('age', 40),
            gender=user_input.get('gender'),
            bmi=user_input.get('bmi'),
            glucose=user_input.get('glucose'),
            sodium_mg_day=user_input.get('sodium_mg_day'),
            sugar_g_day=user_input.get('sugar_g_day'),
            activity_minutes_week=user_input.get('activity_minutes_week'),
            smoking_status=user_input.get('smoking_status'),
            fiber_g_day=user_input.get('fiber_g_day'),
            protein_g_day=user_input.get('protein_g_day'),
            alcohol_g_day=user_input.get('alcohol_g_day'),
            systolic_bp=user_input.get('systolic_bp'),
            diastolic_bp=user_input.get('diastolic_bp')
        )
        
        lifestyle_context = self.lifestyle_recommender.build_lifestyle_context(
            predicted_disease, 
            user_metrics
        )
        
        # Build enhanced prompt
        prompt = self._build_recommendation_prompt(
            predicted_disease, confidence, user_input, profile, comparisons, top_3, lifestyle_context
        )
        
        # Call Hugging Face API using new chat completions format
        try:
            print(f"Calling Hugging Face API ({self.hf_model})...")
            
            # Use the new chat completions API
            try:
                from huggingface_hub import InferenceClient
                
                client = InferenceClient(api_key=self.hf_api_key)
                
                # Format as chat message
                completion = client.chat.completions.create(
                    model=self.hf_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a health lifestyle assistant providing evidence-based recommendations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.7,
                    top_p=0.9
                )
                
                recommendations = completion.choices[0].message.content
                print("✓ Recommendations generated successfully (via chat completions API)")
                
            except Exception as e:
                # Fallback to direct requests
                print(f"⚠️  Chat API failed: {e}")
                print("Trying direct API call...")
                import requests
                
                headers = {
                    "Authorization": f"Bearer {self.hf_api_key}",
                    "Content-Type": "application/json"
                }
                
                api_url = "https://router.huggingface.co/v1/chat/completions"
                
                payload = {
                    "model": self.hf_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a health lifestyle assistant providing evidence-based recommendations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
                
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=180
                )
                
                if response.status_code != 200:
                    raise Exception(f"API returned {response.status_code}: {response.text}")
                
                result = response.json()
                recommendations = result["choices"][0]["message"]["content"]
                print("✓ Recommendations generated successfully (via direct API)")
            
        except Exception as e:
            print(f"⚠️  LLM API error: {e}")
            print("Using fallback rule-based recommendations")
            recommendations = self._generate_fallback_recommendations(predicted_disease, profile, comparisons, top_3, lifestyle_context, user_input)
        
        return recommendations
    
    def _build_recommendation_prompt(self, 
                                    predicted_disease: str,
                                    confidence: float,
                                    user_input: Dict,
                                    profile: Dict,
                                    comparisons: Dict,
                                    top_3: Dict,
                                    lifestyle_context: Dict) -> str:
        """Enhanced prompt with cultural diet plans and workout options"""
        
        # Build risk table from top 3 predictions
        risk_table = [
            {"disease": disease, "prob": float(prob)} 
            for disease, prob in top_3.items()
        ]
        risk_json = json.dumps(risk_table, indent=2)
        ctx_json = json.dumps(lifestyle_context, indent=2)
        
        # Get user preferences
        ethnicity = user_input.get('ethnicity', 'General')
        wants_diet_plan = user_input.get('wants_diet_plan', False)
        wants_workout_plan = user_input.get('wants_workout_plan', False)
        dietary_restrictions = user_input.get('dietary_restrictions', [])
        
        system_prompt = f"""You are a health lifestyle assistant specializing in culturally-adapted wellness guidance.
You DO NOT diagnose or prescribe medications.
You ONLY provide lifestyle recommendations (diet, activity, smoking, alcohol, sleep, etc.).

You will receive:
- The user's free-text query (symptoms)
- A "risk_table" with model-estimated disease probabilities (NOT diagnoses)
- A "context" JSON with:
    - disease (string): top predicted disease name (for reference)
    - age, age_band
    - metrics (dict): keys like sugar, sodium, activity, bmi, fiber, protein, alcohol, smoking_status
    - notes (list): tags like "sugar_high", "activity_low", "smoker", etc.

Each metric may include fields like:
    "user": user's value (float),
    "population_mean": NHANES mean or null,
    "guideline" / "guideline_min" / "guideline_max": thresholds,
    "intermediate_target": realistic next-step target,
    "unit": unit string, e.g. "g/day", "mg/day", "min/week",
    "status": "high" | "low" | "ok" | "unknown" | "very_high" | etc.

CULTURAL CONTEXT:
- User's ethnicity/region: {ethnicity}
- Wants detailed diet plan: {wants_diet_plan}
- Wants workout plan: {wants_workout_plan}
- Dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}

CRITICAL INSTRUCTIONS:

1) Your response MUST have two sections with these EXACT headers:

### 1. Disease risk estimation (not a diagnosis)

- Summarize the risk_table in 2–4 sentences.
- Mention each disease and its probability.
- Emphasize that these are probability scores from a machine learning model, not a formal diagnosis.
- Encourage the user to see a clinician, especially for serious symptoms (like chest pain, shortness of breath, etc.).

### 2. Lifestyle review and recommendations

Using context.metrics, provide specific guidance for each metric with issues:

- For "sugar":
    * If status == "high":
        - Mention user's sugar (user), guideline, intermediate_target.
        - Suggest specific, realistic changes (e.g., reduce sugary drinks, move toward target).
        - If wants_diet_plan: Provide culturally-appropriate meal examples that are lower in sugar

- For "sodium":
    * If status == "high":
        - Use user, guideline, intermediate_target.
        - Suggest reducing processed/packaged foods, cooking with less salt.
        - If wants_diet_plan: Give specific low-sodium recipes/meals from their cuisine

- For "activity":
    * If status == "low":
        - Mention user vs guideline_min.
        - Use intermediate_target as a first goal.
        - If wants_workout_plan: Provide a SPECIFIC weekly workout schedule with:
            * Days of week (e.g., Mon/Wed/Fri)
            * Type of exercise (walking, swimming, cycling, etc.)
            * Duration per session
            * Progression plan (how to increase over 4-8 weeks)
            * Cultural considerations (e.g., gym vs home vs outdoor preferences)
    * If user >= guideline_min:
        - Acknowledge they meet or exceed guidelines.
        - DO NOT tell them to exercise more by default; say "maintain this level" instead.

- For "bmi":
    * If status is "overweight" or "obese" or "underweight":
        - Briefly explain this in simple language and connect it to lifestyle, not body-shaming.
        - Tie back to diet/activity changes you already recommend.

- For "fiber":
    * If status == "low":
        - Use guideline_min and intermediate_target.
        - Suggest concrete foods (beans, lentils, oats, fruit, vegetables).
        - If wants_diet_plan: Provide culturally-appropriate high-fiber meal ideas

- For "protein":
    * Do NOT act like there is a strict "maximum".
    * Explain that ~0.8 g/kg body weight is baseline, 1.2–2.0 g/kg for active people.
    * If status == "low", gently encourage more protein (using intermediate_target).
    * If status == "very_high", simply say they may not need that much, but avoid alarming language.
    * If wants_diet_plan: Suggest protein-rich foods from their cuisine

- For "alcohol":
    * If status == "high":
        - Use user, guideline_max, intermediate_target.
        - Suggest specific ways to cut back (e.g., drink-free days).

- For "smoking_status":
    * If user == "current":
        - Encourage cutting down and quitting as a major health win.
    * If "none" or "former":
        - Acknowledge this as a protective factor; do NOT tell them to quit.

CULTURAL DIET PLAN GUIDELINES (if wants_diet_plan == True):

For **American/Western** cuisine:
- Breakfast: Oatmeal with berries, Greek yogurt, whole grain toast
- Lunch: Grilled chicken salad, turkey sandwich on whole wheat
- Dinner: Baked salmon with roasted vegetables, lean beef with quinoa

For **Asian** (Chinese/Japanese/Korean) cuisine:
- Focus on: Brown rice, miso soup, steamed vegetables, tofu, fish
- Example: Steamed fish with bok choy, brown rice, edamame

For **Indian/South Asian** cuisine:
- Focus on: Lentil dal, roti (whole wheat), vegetables curries, brown rice
- Example: Chana masala, palak paneer, brown rice, raita
- Reduce: Oil in curries, white rice, fried foods

For **Mexican/Latin American** cuisine:
- Focus on: Black beans, grilled chicken/fish, corn tortillas, salsa, avocado
- Example: Chicken fajitas with corn tortillas, black beans, guacamole
- Reduce: Cheese, sour cream, fried items

For **Mediterranean** cuisine:
- Focus on: Olive oil, fish, vegetables, whole grains, legumes
- Example: Grilled fish with olive oil, Greek salad, whole grain pita

WORKOUT PLAN GUIDELINES (if wants_workout_plan == True):

Provide a STRUCTURED weekly plan like this format:

**Week 1-2 (Getting Started - [intermediate_target] minutes/week):**
- Monday: 15-min brisk walk
- Wednesday: 15-min brisk walk  
- Friday: 20-min brisk walk
- Total: ~50 minutes/week

**Week 3-4 (Building Up - 75 minutes/week):**
- Monday: 20-min walk
- Wednesday: 25-min walk + 5-min stretching
- Friday: 20-min walk
- Saturday: 10-min light activity
- Total: ~80 minutes/week

**Week 5-8 (Reaching Goal - {lifestyle_context.get('metrics', {}).get('activity', {}).get('guideline_min', 150)} minutes/week):**
- [Specific progressive plan to reach guideline]

Include:
- Specific exercises suitable for their age and condition
- Home vs gym options
- Low-impact options if needed (for elderly, joint issues)
- Cultural context (e.g., walking in community parks, group activities common in their culture)

STYLE:
- Use numeric fields from JSON directly; do not invent new numbers.
- Make 2–5 specific, actionable recommendations overall, not 10+.
- If diet/workout plan requested, make it DETAILED and SPECIFIC.
- Keep tone calm, supportive, and non-judgmental.
- End with: "This is general lifestyle guidance and not medical advice; please talk to a clinician about any concerning symptoms."
"""

        user_prompt = f"""
User's original query (symptoms):
{user_input.get('symptoms_text', 'No symptoms provided')}

Model-estimated disease risk table (NOT diagnoses):
{risk_json}

NHANES + guideline context (JSON):
{ctx_json}

User preferences:
- Ethnicity/Region: {ethnicity}
- Wants detailed diet plan: {wants_diet_plan}
- Wants workout plan: {wants_workout_plan}
- Dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}

Please follow the required structure:
1. "### 1. Disease risk estimation (not a diagnosis)"
2. "### 2. Lifestyle review and recommendations"

{'Include a DETAILED, culturally-appropriate diet plan with specific meals.' if wants_diet_plan else ''}
{'Include a SPECIFIC progressive workout schedule (4-8 weeks).' if wants_workout_plan else ''}
"""
        
        return system_prompt + "\n\n" + user_prompt
    
    def _generate_fallback_recommendations(self, 
                                          disease: str, 
                                          profile: Dict,
                                          comparisons: Dict,
                                          top_3: Dict,
                                          lifestyle_context: Dict,
                                          user_input: Dict) -> str:
        """Enhanced rule-based recommendations matching LLM output quality"""
        
        recs = f"### 1. Disease risk estimation (not a diagnosis)\n\n"
        
        # Risk estimation section
        recs += "Based on the information provided, a machine learning model has estimated the probability of certain conditions. "
        
        # List top 3 predictions
        predictions_text = []
        for disease_name, prob in top_3.items():
            predictions_text.append(f"{prob*100:.1f}% probability for {disease_name}")
        
        recs += f"The top results are {', '.join(predictions_text)}. "
        recs += "It is very important to understand that **these are probability scores from a model and not a medical diagnosis.**\n\n"
        
        # Check for serious symptoms
        symptoms = user_input.get('symptoms_text', '').lower()
        serious_symptoms = ['chest pain', 'shortness of breath', 'difficulty breathing', 'severe headache', 'confusion']
        if any(symptom in symptoms for symptom in serious_symptoms):
            recs += "Given your symptoms, it is crucial that you speak with a healthcare provider for a proper evaluation.\n\n"
        else:
            recs += "Please consult with a healthcare provider for a proper medical evaluation.\n\n"
        
        # Lifestyle recommendations section
        recs += "### 2. Lifestyle review and recommendations\n\n"
        recs += "Here is a review of your lifestyle factors and some suggestions for potential improvements.\n\n"
        
        # Process each metric from lifestyle context
        metrics = lifestyle_context.get('metrics', {})
        
        # Physical Activity
        if 'activity' in metrics:
            activity = metrics['activity']
            if activity.get('status') == 'low':
                user_val = activity['user']
                guideline = activity['guideline_min']
                target = activity.get('intermediate_target')
                recs += f"* **Physical Activity**: Your current activity level is {user_val} minutes per week, "
                recs += f"while guidelines suggest a minimum of {guideline} minutes for general health. "
                if target:
                    recs += f"A great first step could be to aim for an initial target of {target} minutes per week. "
                recs += "This could be achieved with something as simple as a 15-minute brisk walk on most days of the week.\n\n"
            elif activity.get('status') == 'ok':
                recs += f"* **Physical Activity**: You're meeting the recommended {activity['guideline_min']} minutes per week. Great job maintaining this healthy habit!\n\n"
        
        # Sugar
        if 'sugar' in metrics:
            sugar = metrics['sugar']
            if sugar.get('status') == 'high':
                user_val = sugar['user']
                guideline = sugar['guideline']
                target = sugar.get('intermediate_target')
                recs += f"* **Added Sugar**: Your estimated sugar intake is {user_val} g/day, "
                recs += f"which is above the general guideline of {guideline} g/day. "
                if target:
                    recs += f"A realistic first goal could be to reduce this to around {target} g/day. "
                recs += "One of the most effective ways to do this is by cutting back on sugary drinks like soda, sweetened teas, and juices.\n\n"
        
        # Fiber
        if 'fiber' in metrics:
            fiber = metrics['fiber']
            if fiber.get('status') == 'low':
                user_val = fiber['user']
                guideline = fiber['guideline_min']
                target = fiber.get('intermediate_target')
                recs += f"* **Dietary Fiber**: Your current fiber intake is around {user_val} g/day, "
                recs += f"below the recommended minimum of {guideline} g/day. "
                if target:
                    recs += f"To start, you could aim for an intermediate target of {target} g/day. "
                recs += "You can increase your fiber intake by adding more whole foods to your diet, such as beans, lentils, oats, fruits (like apples and berries), and vegetables.\n\n"
        
        # Sodium
        if 'sodium' in metrics:
            sodium = metrics['sodium']
            if sodium.get('status') == 'high':
                user_val = sodium['user']
                guideline = sodium['guideline']
                target = sodium.get('intermediate_target')
                recs += f"* **Sodium Intake**: Your sodium consumption is approximately {user_val} mg/day, "
                recs += f"which exceeds the recommended limit of {guideline} mg/day. "
                if target:
                    recs += f"A realistic first target would be {target} mg/day. "
                recs += "Consider reducing processed and packaged foods, and use less salt when cooking.\n\n"
        
        # Alcohol
        if 'alcohol' in metrics:
            alcohol = metrics['alcohol']
            if alcohol.get('status') == 'high':
                user_val = alcohol['user']
                guideline = alcohol['guideline_max']
                recs += f"* **Alcohol Intake**: Your alcohol consumption is about {user_val} g/day, "
                recs += f"which is above the guideline maximum of {guideline} g/day. "
                recs += "Consider introducing one or two alcohol-free days per week or choosing smaller drink sizes to help bring your average intake within the recommended limits.\n\n"
        
        # Protein
        if 'protein' in metrics:
            protein = metrics['protein']
            if protein.get('status') == 'low':
                user_val = protein['user']
                guideline = protein['guideline_min']
                target = protein.get('intermediate_target')
                recs += f"* **Protein Intake**: Your protein intake is {user_val} g/day. "
                recs += f"The baseline recommendation is around {guideline} g/day (about 0.8g per kg of body weight). "
                if target:
                    recs += f"Consider aiming for {target} g/day as a first step. "
                recs += "Good protein sources include lean meats, fish, eggs, legumes, and dairy products.\n\n"
        
        # BMI
        if 'bmi' in metrics:
            bmi = metrics['bmi']
            user_val = bmi['user']
            status = bmi['status']
            if status in ['overweight', 'obese']:
                recs += f"* **Body Mass Index (BMI)**: Your BMI of {user_val} is in the {status} category. "
                recs += "The dietary and activity changes suggested above can contribute positively to achieving a healthier weight over time. "
            elif status == 'healthy_range':
                recs += f"* **Body Mass Index (BMI)**: Your BMI of {user_val} is in the healthy range. Keep up the good work!\n\n"
        
        # Glucose
        if 'glucose' in metrics:
            glucose = metrics['glucose']
            if glucose.get('status') in ['prediabetic_range', 'diabetic_range']:
                user_val = glucose['user']
                status = glucose['status']
                recs += f"* **Blood Glucose**: Your fasting glucose of {user_val} mg/dL is in the {status.replace('_', ' ')}. "
                recs += "The dietary changes above (reducing sugar, increasing fiber) along with regular physical activity can help manage blood sugar levels. "
                recs += "Regular monitoring and consultation with your healthcare provider are important.\n\n"
        
        # Blood Pressure
        if 'blood_pressure' in metrics:
            bp = metrics['blood_pressure']
            if bp.get('status') != 'normal':
                systolic = bp['user_systolic']
                diastolic = bp.get('user_diastolic', 80)
                status = bp['status']
                recs += f"* **Blood Pressure**: Your reading of {systolic}/{diastolic} mmHg indicates {status.replace('_', ' ')}. "
                recs += "Reducing sodium intake, maintaining a healthy weight, regular exercise, and limiting alcohol can help manage blood pressure. "
                recs += "Regular monitoring is important.\n\n"
        
        # Smoking
        if 'smoking_status' in metrics:
            smoking = metrics['smoking_status']
            status = smoking['user']
            if status == 'none' or status == 'former':
                recs += f"* **Smoking Status**: It's great that you are a non-smoker"
                if status == 'former':
                    recs += " (former smoker)"
                recs += ", which is a major positive for your overall health.\n\n"
            elif status == 'current':
                recs += "* **Smoking**: Quitting smoking is one of the most impactful changes you can make for your health. "
                recs += "Consider speaking with your healthcare provider about smoking cessation programs and resources.\n\n"
        
        # Closing
        recs += "\nThis is general lifestyle guidance and not medical advice; please talk to a clinician about any concerning symptoms.\n"
        
        return recs
    
    def generate_comprehensive_report(self, user_input: Dict) -> Dict:
        """
        🎯 MAIN METHOD: Complete pipeline
        
        Example user_input:
        {
            'age': 55,
            'gender': 'M',
            'symptoms_text': 'increased thirst, frequent urination, fatigue',
            'glucose': 145,
            'systolic_bp': 142,
            'diastolic_bp': 88,
            'creatinine': 1.3,
            'hematocrit': 38.5,
            'bmi': 32.5
        }
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE HEALTH ANALYSIS & RECOMMENDATIONS")
        print("="*70)
        print(f"\nPatient: {user_input.get('gender', 'Unknown')}, Age {user_input.get('age', 'Unknown')}")
        
        # Step 1: Disease Prediction
        predicted_disease, confidence, probabilities, top_3 = self.predict_disease(user_input)
        
        # Step 2: Get NHANES Profile
        profile = self.get_nhanes_profile(predicted_disease)
        
        # Step 3: Compare to Population
        comparisons = self.compare_user_to_population(user_input, profile)
        
        # Step 4: Generate Recommendations
        recommendations = self.generate_llm_recommendations(
            predicted_disease, confidence, user_input, profile, comparisons, top_3
        )
        
        # Compile report
        report = {
            'metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_version': 'Random Forest (F1=0.8776, AUROC=0.9847)',
                'nhanes_version': '2021-2023'
            },
            'patient_info': {
                'age': user_input.get('age'),
                'gender': user_input.get('gender'),
                'bmi': user_input.get('bmi'),
                'symptoms': user_input.get('symptoms_text', user_input.get('symptoms', []))
            },
            'prediction': {
                'disease': predicted_disease,
                'confidence': float(confidence),
                'top_3_predictions': top_3,
                'model_used': 'Random Forest'
            },
            'population_analysis': {
                'nhanes_profile': profile.get('disease', 'Unknown'),
                'sample_size': profile.get('sample_size', 0),
                'prevalence': profile.get('nhanes_prevalence', 'N/A'),
                'comparisons': comparisons
            },
            'recommendations': recommendations
        }
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        
        return report
    
    def save_report(self, report: Dict, output_dir: str = 'reports/patient_reports') -> Path:
        """Save report to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        disease = report['prediction']['disease'].replace(' ', '_')
        filename = f"health_report_{disease}_{timestamp}.json"
        
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to: {filepath}")
        return filepath
    
    def print_report_summary(self, report: Dict):
        """Print formatted report summary"""
        print("\n" + "="*70)
        print("PATIENT HEALTH REPORT SUMMARY")
        print("="*70)
        
        # Patient info
        print(f"\n📋 Patient Information:")
        print(f"   Age: {report['patient_info']['age']} years")
        print(f"   Gender: {report['patient_info']['gender']}")
        if report['patient_info'].get('bmi'):
            print(f"   BMI: {report['patient_info']['bmi']} kg/m²")
        
        # Prediction
        print(f"\n🔬 Analysis Results:")
        print(f"   Predicted Condition: {report['prediction']['disease']}")
        print(f"   Confidence: {report['prediction']['confidence']*100:.1f}%")
        print(f"   Model: {report['metadata']['model_version']}")
        
        print(f"\n   Top 3 Predictions:")
        for disease, prob in report['prediction']['top_3_predictions'].items():
            print(f"     • {disease}: {prob*100:.1f}%")
        
        # Population comparison
        print(f"\n📊 Population Comparison:")
        print(f"   Reference: {report['population_analysis']['sample_size']:,} NHANES participants")
        print(f"   Prevalence: {report['population_analysis']['prevalence']}")
        
        if report['population_analysis']['comparisons']:
            print(f"\n   Clinical Markers:")
            for marker, data in report['population_analysis']['comparisons'].items():
                status = data.get('status', 'unknown')
                print(f"     • {marker.replace('_', ' ').title()}: {status}")
        
        # Recommendations preview
        print(f"\n💡 Recommendations:")
        recommendations = report['recommendations']
        if isinstance(recommendations, str):
            lines = recommendations.split('\n')[:15]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
            if len(recommendations.split('\n')) > 15:
                print("\n   [See full report for complete recommendations]")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_diabetes_case():
    """Example: Diabetes patient with lifestyle metrics"""
    
    print("\n" + "="*70)
    print("EXAMPLE: DIABETES CASE")
    print("="*70)
    
    # Initialize system with Hugging Face API
    import os
    hf_api_key = os.getenv('HF_API_KEY', None)
    
    recommender = ComprehensiveRecommendationSystem(
        model_dir='models_fixed',
        profiles_path='models_fixed/comprehensive_disease_profiles.json',
        use_llm=True if hf_api_key else False,
        hf_api_key=hf_api_key,
        hf_model='meta-llama/Llama-3.3-70B-Instruct:groq'  # Updated to use new API format
    )
    
    # Patient data with lifestyle metrics
    patient_data = {
        # Basic info
        'age': 55,
        'gender': 'M',
        'symptoms_text': 'increased thirst frequent urination fatigue blurred vision',
        
        # Clinical measurements
        'glucose': 148,
        'systolic_bp': 145,
        'diastolic_bp': 92,
        'hematocrit': 38.5,
        'creatinine': 1.1,
        'sodium': 140,
        'potassium': 4.2,
        'bmi': 32.5,
        
        # NEW: Lifestyle metrics for enhanced recommendations
        'sugar_g_day': 120,  # High added sugar
        'sodium_mg_day': 4000,  # High sodium
        'activity_minutes_week': 20,  # Low activity
        'fiber_g_day': 12,  # Low fiber
        'protein_g_day': 95,  # Adequate protein
        'alcohol_g_day': 16,  # Slightly high alcohol
        'smoking_status': 'none'  # Non-smoker
    }
    
    # Generate report
    report = recommender.generate_comprehensive_report(patient_data)
    
    # Print summary
    recommender.print_report_summary(report)
    
    # Save
    filepath = recommender.save_report(report)
    
    return report


def example_hypertension_case():
    """Example: Hypertension patient with lifestyle metrics"""
    
    print("\n" + "="*70)
    print("EXAMPLE: HYPERTENSION CASE")
    print("="*70)
    
    import os
    hf_api_key = os.getenv('HF_API_KEY', None)
    
    recommender = ComprehensiveRecommendationSystem(
        model_dir='models_fixed',
        profiles_path='models_fixed/comprehensive_disease_profiles.json',
        use_llm=True if hf_api_key else False,
        hf_api_key=hf_api_key
    )
    
    patient_data = {
        # Basic info
        'age': 62,
        'gender': 'F',
        'symptoms_text': 'headache chest pain dizziness difficulty concentrating',
        
        # Clinical measurements
        'systolic_bp': 152,
        'diastolic_bp': 96,
        'glucose': 105,
        'hematocrit': 42.0,
        'sodium': 142,
        'bmi': 28.5,
        
        # NEW: Lifestyle metrics for enhanced recommendations
        'sugar_g_day': 75,  # Added sugar intake
        'sodium_mg_day': 3800,  # Daily sodium
        'activity_minutes_week': 45,  # Physical activity
        'fiber_g_day': 14,  # Dietary fiber
        'protein_g_day': 65,  # Protein intake
        'alcohol_g_day': 8,  # Alcohol consumption
        'smoking_status': 'former'  # Smoking status
    }
    
    report = recommender.generate_comprehensive_report(patient_data)
    recommender.print_report_summary(report)
    filepath = recommender.save_report(report)
    
    return report


def example_kidney_failure_case():
    """Example: Kidney failure patient with lifestyle metrics"""
    
    print("\n" + "="*70)
    print("EXAMPLE: KIDNEY FAILURE CASE")
    print("="*70)
    
    import os
    hf_api_key = os.getenv('HF_API_KEY', None)
    
    recommender = ComprehensiveRecommendationSystem(
        model_dir='models_fixed',
        profiles_path='models_fixed/comprehensive_disease_profiles.json',
        use_llm=True if hf_api_key else False,
        hf_api_key=hf_api_key
    )
    
    patient_data = {
        # Basic info
        'age': 68,
        'gender': 'M',
        'symptoms_text': 'fatigue nausea metallic taste decreased urine output swelling legs',
        
        # Clinical measurements
        'creatinine': 2.8,
        'urea_nitrogen': 45,
        'glucose': 110,
        'systolic_bp': 155,
        'diastolic_bp': 92,
        'hematocrit': 32.0,
        'potassium': 5.2,
        'bmi': 29.0,
        
        # NEW: Lifestyle metrics
        'sugar_g_day': 65,
        'sodium_mg_day': 4200,  # Very high - concern for kidney disease
        'activity_minutes_week': 30,
        'fiber_g_day': 16,
        'protein_g_day': 110,  # High - concern for kidney disease
        'alcohol_g_day': 5,
        'smoking_status': 'former'
    }
    
    report = recommender.generate_comprehensive_report(patient_data)
    recommender.print_report_summary(report)
    filepath = recommender.save_report(report)
    
    return report


if __name__ == "__main__":
    # Run all examples
    print("\n" + "="*70)
    print("RUNNING EXAMPLE CASES")
    print("="*70)
    print("\nNote: Set HF_TOKEN environment variable to enable LLM recommendations")
    print("For now, using enhanced rule-based recommendations\n")
    
    print("\n\n")
    report1 = example_diabetes_case()
    
    print("\n\n")
    report2 = example_hypertension_case()
    
    print("\n\n")
    report3 = example_kidney_failure_case()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nGenerated 3 patient reports in: reports/patient_reports/")
    print("\nTo enable LLM recommendations:")
    print("  PowerShell: $env:HF_TOKEN = 'your_huggingface_token_here'")
    print("  Then create a new token at: https://huggingface.co/settings/tokens")
    print("  Enable 'Make calls to Inference Providers' permission")