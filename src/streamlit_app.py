import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent))

# Load environment variables
try:
    from load_env import load_env, get_config
    load_env()
    config = get_config()
except ImportError:
    print("⚠️  load_env.py not found - using environment variables directly")
    config = {
        'hf_token': os.getenv('HF_TOKEN') or os.getenv('HF_API_KEY'),
        'hf_model': 'meta-llama/Llama-3.3-70B-Instruct:groq',
        'use_llm': True,
        'model_dir': 'models_fixed',
        'profiles_path': 'models_fixed/comprehensive_disease_profiles.json'
    }

from recommendation_engine import ComprehensiveRecommendationSystem

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Patient Health Analysis & Wellness Guidance",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2874A6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'report' not in st.session_state:
    st.session_state.report = None

if 'system' not in st.session_state:
    with st.spinner("🔄 Initializing recommendation system..."):
        try:
            st.session_state.system = ComprehensiveRecommendationSystem(
                model_dir=config['model_dir'],
                profiles_path=config['profiles_path'],
                use_llm=config['use_llm'] and config['hf_token'] is not None,
                hf_api_key=config['hf_token'],
                hf_model=config['hf_model']
            )
            st.session_state.system_loaded = True
            
            # Show status message
            if config['hf_token']:
                st.success(f"✅ LLM enabled with {config['hf_model']}")
            else:
                st.warning("⚠️  No HF_TOKEN found - using enhanced rule-based recommendations")
                
        except Exception as e:
            st.error(f"❌ Error loading system: {e}")
            st.session_state.system_loaded = False

if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        🏥 Patient Health Analysis & Wellness Guidance Tool
    </div>
    """, unsafe_allow_html=True)
    
    # Subtitle
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔬 MIMIC-IV Disease Prediction**")
        st.caption("Random Forest (F1: 87.8%, AUROC: 98.5%)")
    with col2:
        st.markdown("**📊 NHANES 2021-2023 Population Data**")
        st.caption("Evidence-based lifestyle guidance")
    with col3:
        llm_status = "✅ Enabled" if st.session_state.system.use_llm else "⚠️ Rule-based"
        st.markdown(f"**🤖 AI Recommendations**")
        st.caption(llm_status)
    
    st.markdown("---")
    
    # Check if system loaded
    if not st.session_state.system_loaded:
        st.error("❌ System failed to load. Please check model files.")
        return
    
    # ========================================================================
    # SIDEBAR - PATIENT INPUT
    # ========================================================================
    
    with st.sidebar:
        st.header("📋 Patient Information")
        
        with st.expander("ℹ️ How to Use", expanded=False):
            st.markdown("""
            **Step 1:** Enter demographics & symptoms  
            **Step 2:** Add clinical measurements  
            **Step 3:** Add lifestyle metrics (optional)  
            **Step 4:** Click "🔬 Analyze Health"  
            """)
        
        st.markdown("---")
        
        # Demographics
        st.subheader("👤 Demographics")
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_code = 'M' if gender == 'Male' else 'F'
        
        # BMI
        st.markdown("**BMI Calculator**")
        col1, col2 = st.columns(2)
        with col1:
            height_cm = st.number_input("Height (cm)", 50, 250, 170)
        with col2:
            weight_kg = st.number_input("Weight (kg)", 20, 300, 70)
        
        bmi = round(weight_kg / ((height_cm/100) ** 2), 1)
        st.info(f"📊 BMI: **{bmi}** kg/m²")
        
        st.markdown("---")
        
        # Symptoms
        st.subheader("🩺 Symptoms")
        symptoms_text = st.text_area(
            "Describe symptoms:",
            placeholder="Example: chest pain, shortness of breath, fatigue",
            height=100
        )
        
        st.markdown("---")
        
        # Clinical Measurements
        st.subheader("🧪 Clinical Measurements")
        
        with st.expander("Blood Work", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                glucose = st.number_input("Glucose (mg/dL)", 0, 500, 100)
                hematocrit = st.number_input("Hematocrit (%)", 0.0, 60.0, 40.0, 0.1)
                creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 15.0, 1.0, 0.1)
            
            with col2:
                sodium = st.number_input("Sodium (mEq/L)", 0, 200, 140)
                potassium = st.number_input("Potassium (mEq/L)", 0.0, 10.0, 4.0, 0.1)
                urea_nitrogen = st.number_input("BUN (mg/dL)", 0, 200, 15)
        
        with st.expander("Blood Pressure", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                systolic_bp = st.number_input("Systolic (mmHg)", 70, 250, 120)
            with col2:
                diastolic_bp = st.number_input("Diastolic (mmHg)", 40, 150, 80)
        
        st.markdown("---")
        
        # NEW: Lifestyle Metrics
        st.subheader("🏃 Lifestyle Metrics")
        st.caption("Optional - for enhanced recommendations")
        
        with st.expander("Dietary Intake", expanded=False):
            sugar_g_day = st.number_input(
                "Added Sugar (g/day)", 
                0, 200, 50,
                help="Average daily added sugar intake"
            )
            sodium_mg_day = st.number_input(
                "Sodium (mg/day)", 
                0, 8000, 2300,
                help="Average daily sodium intake"
            )
            fiber_g_day = st.number_input(
                "Fiber (g/day)", 
                0, 100, 25,
                help="Average daily fiber intake"
            )
            protein_g_day = st.number_input(
                "Protein (g/day)", 
                0, 300, 75,
                help="Average daily protein intake"
            )
            alcohol_g_day = st.number_input(
                "Alcohol (g/day)", 
                0, 100, 0,
                help="Average daily alcohol consumption"
            )
        
        with st.expander("Physical Activity", expanded=False):
            activity_minutes_week = st.number_input(
                "Exercise (min/week)", 
                0, 1000, 150,
                help="Total weekly moderate-intensity activity"
            )
        
        with st.expander("Other Habits", expanded=False):
            smoking_status = st.selectbox(
                "Smoking Status",
                ["none", "former", "current"],
                help="Current smoking status"
            )
        
        st.markdown("---")
        
        # NEW: Personalization Options
        st.subheader("🌍 Personalization")
        st.caption("Get culturally-appropriate recommendations")
        
        ethnicity = st.selectbox(
            "Cultural/Regional Background",
            ["General/Western", "Asian (Chinese/Japanese/Korean)", "Indian/South Asian", 
             "Mexican/Latin American", "Mediterranean", "Middle Eastern", "African"],
            help="Select for culturally-appropriate meal suggestions"
        )
        
        st.markdown("**Would you like detailed plans?**")
        wants_diet_plan = st.checkbox(
            "📋 Generate detailed meal plan",
            help="Get specific breakfast, lunch, dinner examples from your cuisine"
        )
        
        wants_workout_plan = st.checkbox(
            "🏋️ Generate progressive workout schedule",
            help="Get a detailed 4-8 week exercise plan"
        )
        
        if wants_diet_plan or wants_workout_plan:
            dietary_restrictions = st.multiselect(
                "Dietary restrictions (if any):",
                ["Vegetarian", "Vegan", "Halal", "Kosher", "Gluten-free", "Dairy-free", "Nut allergies"],
                help="Select any dietary restrictions"
            )
        else:
            dietary_restrictions = []
        
        st.markdown("---")
        
        # Analyze button
        analyze_button = st.button("🔬 Analyze Health", type="primary", use_container_width=True)
        
        # Clear button
        if st.session_state.report:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.report = None
                st.rerun()
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    if analyze_button:
        if not symptoms_text.strip():
            st.warning("⚠️ Please enter symptoms before analyzing.")
        else:
            # Prepare comprehensive user input
            user_input = {
                # Demographics
                'age': age,
                'gender': gender_code,
                'bmi': bmi,
                'symptoms_text': symptoms_text,
                
                # Clinical measurements
                'glucose': glucose if glucose > 0 else None,
                'systolic_bp': systolic_bp if systolic_bp > 0 else None,
                'diastolic_bp': diastolic_bp if diastolic_bp > 0 else None,
                'hematocrit': hematocrit if hematocrit > 0 else None,
                'creatinine': creatinine if creatinine > 0 else None,
                'sodium': sodium if sodium > 0 else None,
                'potassium': potassium if potassium > 0 else None,
                'urea_nitrogen': urea_nitrogen if urea_nitrogen > 0 else None,
                
                # Lifestyle metrics
                'sugar_g_day': sugar_g_day if sugar_g_day > 0 else None,
                'sodium_mg_day': sodium_mg_day if sodium_mg_day > 0 else None,
                'fiber_g_day': fiber_g_day if fiber_g_day > 0 else None,
                'protein_g_day': protein_g_day if protein_g_day > 0 else None,
                'alcohol_g_day': alcohol_g_day if alcohol_g_day > 0 else None,
                'activity_minutes_week': activity_minutes_week if activity_minutes_week > 0 else None,
                'smoking_status': smoking_status if smoking_status != 'none' else None,
                
                # NEW: Personalization preferences
                'ethnicity': ethnicity,
                'wants_diet_plan': wants_diet_plan,
                'wants_workout_plan': wants_workout_plan,
                'dietary_restrictions': dietary_restrictions
            }
            
            # Generate report
            with st.spinner("🔄 Analyzing health data... This may take 30-60 seconds..."):
                try:
                    report = st.session_state.system.generate_comprehensive_report(user_input)
                    st.session_state.report = report
                    
                    # Add to history
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'disease': report['prediction']['disease'],
                        'confidence': report['prediction']['confidence'],
                        'age': age,
                        'gender': gender
                    })
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    st.exception(e)
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    if st.session_state.report:
        report = st.session_state.report
        
        # Create tabs
        tabs = st.tabs([
            "🔬 Prediction Results", 
            "📊 Population Comparison", 
            "💡 Recommendations",
            "📈 Visualizations",
            "📄 Full Report"
        ])
        
        # TAB 1: PREDICTION RESULTS
        with tabs[0]:
            display_prediction_results(report)
        
        # TAB 2: POPULATION COMPARISON
        with tabs[1]:
            display_population_comparison(report)
        
        # TAB 3: RECOMMENDATIONS
        with tabs[2]:
            display_recommendations(report)
        
        # TAB 4: VISUALIZATIONS
        with tabs[3]:
            display_visualizations(report)
        
        # TAB 5: FULL REPORT
        with tabs[4]:
            display_full_report(report)
    
    else:
        # Welcome screen
        display_welcome_screen()

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_welcome_screen():
    """Display welcome message"""
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to the Patient Health Analysis System</h3>
        <p>This intelligent system provides:</p>
        <ul>
            <li><strong>Disease Risk Prediction</strong> using MIMIC-IV trained AI</li>
            <li><strong>Population Comparison</strong> with NHANES 2021-2023 data</li>
            <li><strong>Personalized Recommendations</strong> with intermediate targets</li>
        </ul>
        <p>👈 Enter patient information in the sidebar to begin analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📚 Sample Cases You Can Try")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🩺 Diabetes Case**
        - Age: 55, Male, BMI: 32.5
        - Symptoms: "increased thirst, frequent urination, fatigue"
        - Glucose: 148 mg/dL
        - Sugar: 120 g/day
        - Activity: 20 min/week
        """)
    
    with col2:
        st.markdown("""
        **🩺 Hypertension Case**
        - Age: 62, Female, BMI: 28.5
        - Symptoms: "headache, chest pain, dizziness"
        - BP: 152/96 mmHg
        - Sodium: 3800 mg/day
        - Activity: 45 min/week
        """)
    
    with col3:
        st.markdown("""
        **🩺 Kidney Disease Case**
        - Age: 68, Male, BMI: 29.0
        - Symptoms: "fatigue, nausea, swelling"
        - Creatinine: 2.8 mg/dL
        - Protein: 110 g/day
        - Sodium: 4200 mg/day
        """)

def display_prediction_results(report):
    """Display disease prediction results"""
    st.header("🔬 Disease Prediction Results")
    
    prediction = report['prediction']
    
    # Main prediction card
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin:0; color: #2874A6;">{prediction['disease']}</h2>
        <h3 style="margin:10px 0 0 0; color: #555;">Confidence: {prediction['confidence']*100:.1f}%</h3>
        <p style="margin:5px 0 0 0; color: #777;">Model: {report['metadata']['model_version']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence indicator
    confidence = prediction['confidence'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Confidence Level", f"{confidence:.1f}%")
    
    with col2:
        st.metric("Model Accuracy", "F1: 87.8%")
    
    st.markdown("---")
    
    # Top 3 predictions chart
    st.subheader("📊 Top 3 Predicted Conditions")
    
    top_3 = prediction['top_3_predictions']
    diseases = list(top_3.keys())
    probabilities = [top_3[d] * 100 for d in diseases]
    
    fig = go.Figure(data=[
        go.Bar(
            y=diseases,
            x=probabilities,
            orientation='h',
            marker=dict(
                color=probabilities,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Probability (%)",
        height=250,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_population_comparison(report):
    """Display NHANES population comparison"""
    st.header("📊 Population Comparison (NHANES 2021-2023)")
    
    pop = report['population_analysis']
    
    st.markdown(f"""
    <div class="info-box">
        <h4>Reference Population</h4>
        <p><strong>Profile:</strong> {pop['nhanes_profile']}</p>
        <p><strong>Sample Size:</strong> {pop['sample_size']:,} participants</p>
        <p><strong>Prevalence:</strong> {pop['prevalence']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    comparisons = pop['comparisons']
    
    if comparisons:
        st.subheader("🎯 Your Values vs Population")
        
        for marker, data in comparisons.items():
            with st.expander(f"📊 {marker.replace('_', ' ').title()}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    user_val = data.get('user_value', data.get('user_systolic', 'N/A'))
                    if 'user_diastolic' in data:
                        user_val = f"{data['user_systolic']}/{data['user_diastolic']}"
                    st.metric("Your Value", user_val)
                
                with col2:
                    st.metric("Population Avg", data.get('population_mean', 'N/A'))
                
                with col3:
                    st.metric("Target", data['target'])
                
                with col4:
                    status = data['status']
                    if status in ['normal', 'healthy_range']:
                        st.success(f"✅ {status.title()}")
                    elif status in ['elevated', 'borderline', 'overweight']:
                        st.warning(f"⚠️ {status.title()}")
                    else:
                        st.error(f"🚨 {status.title()}")
    else:
        st.info("ℹ️ Add lab values for population comparison")

def display_recommendations(report):
    """Display personalized recommendations"""
    st.header("💡 Personalized Health Recommendations")
    
    recommendations = report['recommendations']
    
    # Display formatted recommendations
    st.markdown(recommendations)
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download Recommendations (MD)",
            data=recommendations,
            file_name=f"recommendations_{report['prediction']['disease'].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col2:
        # Convert to text summary
        summary = generate_text_summary(report)
        st.download_button(
            label="📥 Download Full Summary (TXT)",
            data=summary,
            file_name=f"health_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def display_visualizations(report):
    """Display health visualizations"""
    st.header("📈 Health Data Visualizations")
    
    comparisons = report['population_analysis']['comparisons']
    
    if comparisons:
        # Risk radar chart
        st.subheader("🎯 Risk Factor Analysis")
        
        categories = []
        risk_scores = []
        
        for marker, data in comparisons.items():
            categories.append(marker.replace('_', ' ').title())
            
            status = data['status']
            if status in ['normal', 'desirable', 'healthy_range']:
                risk = 20
            elif status in ['elevated', 'borderline', 'overweight', 'prediabetic range']:
                risk = 50
            elif status in ['high', 'obese', 'diabetic range', 'stage 1 hypertension']:
                risk = 75
            else:
                risk = 95
            
            risk_scores.append(risk)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=risk_scores,
            theta=categories,
            fill='toself',
            name='Your Risk Profile',
            line=dict(color='#dc3545', width=2),
            fillcolor='rgba(220, 53, 69, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[30] * len(categories),
            theta=categories,
            fill='toself',
            name='Healthy Range',
            line=dict(color='#28a745', width=2, dash='dot'),
            fillcolor='rgba(40, 167, 69, 0.1)'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Risk Factor Radar (0=Optimal, 100=High Risk)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis history
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📅 Analysis History")
        
        history_df = pd.DataFrame(st.session_state.history)
        
        fig = px.line(
            history_df,
            x='timestamp',
            y='confidence',
            title='Prediction Confidence Over Time',
            markers=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

def display_full_report(report):
    """Display full JSON report"""
    st.header("📄 Complete Health Report")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", f"{report['patient_info']['age']} years")
    with col2:
        st.metric("Gender", report['patient_info']['gender'])
    with col3:
        if report['patient_info'].get('bmi'):
            st.metric("BMI", f"{report['patient_info']['bmi']} kg/m²")
    with col4:
        st.metric("Analysis Date", report['metadata']['timestamp'].split('T')[0])
    
    st.markdown("---")
    
    st.subheader("📋 Detailed Report Data")
    st.json(report)
    
    st.markdown("---")
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=report_json,
            file_name=f"health_report_{report['prediction']['disease'].replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        summary = generate_text_summary(report)
        st.download_button(
            label="📥 Download Summary (TXT)",
            data=summary,
            file_name=f"health_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

def generate_text_summary(report):
    """Generate text summary"""
    summary = "="*70 + "\n"
    summary += "PATIENT HEALTH ANALYSIS REPORT\n"
    summary += "="*70 + "\n\n"
    
    summary += f"Generated: {report['metadata']['timestamp']}\n\n"
    
    summary += "PATIENT INFORMATION\n"
    summary += "-"*70 + "\n"
    summary += f"Age: {report['patient_info']['age']} years\n"
    summary += f"Gender: {report['patient_info']['gender']}\n"
    if report['patient_info'].get('bmi'):
        summary += f"BMI: {report['patient_info']['bmi']} kg/m²\n"
    summary += f"\nSymptoms: {report['patient_info'].get('symptoms', 'N/A')}\n\n"
    
    summary += "PREDICTION RESULTS\n"
    summary += "-"*70 + "\n"
    summary += f"Predicted: {report['prediction']['disease']}\n"
    summary += f"Confidence: {report['prediction']['confidence']*100:.1f}%\n\n"
    
    summary += "Top 3:\n"
    for disease, prob in report['prediction']['top_3_predictions'].items():
        summary += f"  • {disease}: {prob*100:.1f}%\n"
    
    summary += "\n" + "="*70 + "\n"
    summary += "RECOMMENDATIONS\n"
    summary += "="*70 + "\n\n"
    summary += report['recommendations']
    
    return summary

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()