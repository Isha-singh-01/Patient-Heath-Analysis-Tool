import pandas as pd
import numpy as np
from pathlib import Path

class NHANESProcessor:
    """Clean and merge NHANES 2021-2023 datasets"""
    
    def __init__(self, raw_data):
        self.raw_data = raw_data
        
    def select_variables(self):
        """Extract only the variables we need - using ACTUAL 2021-2023 column names"""
        
        print("Selecting relevant variables...")
        
        # ============================================================
        # DEMOGRAPHICS
        # ============================================================
        demo = self.raw_data['demographics'][[
            'SEQN',      # Respondent sequence number (KEY)
            'RIDAGEYR',  # Age in years
            'RIAGENDR',  # Gender (1=Male, 2=Female)
            'RIDRETH3',  # Race/ethnicity
            'DMDEDUC2',  # Education level (adults 20+)
            'INDFMPIR'   # Family income to poverty ratio
        ]].copy()
        print(f"  ✓ Demographics: {len(demo)} rows")
        
        # ============================================================
        # BODY MEASURES
        # ============================================================
        body = self.raw_data['body_measures'][[
            'SEQN',
            'BMXBMI',    # BMI
            'BMXWT',     # Weight (kg)
            'BMXHT',     # Standing height (cm)
            'BMXWAIST',  # Waist circumference (cm)
            'BMXHIP'     # Hip circumference (cm)
        ]].copy()
        print(f"  ✓ Body measures: {len(body)} rows")
        
        # ============================================================
        # BLOOD PRESSURE
        # ============================================================
        bp = self.raw_data['blood_pressure'][[
            'SEQN',
            'BPXOSY1',   # Systolic BP - 1st reading (mmHg)
            'BPXODI1',   # Diastolic BP - 1st reading (mmHg)
            'BPXOSY2',   # Systolic BP - 2nd reading
            'BPXODI2'    # Diastolic BP - 2nd reading
        ]].copy()
        print(f"  ✓ Blood pressure: {len(bp)} rows")
        
        # ============================================================
        # GLUCOSE (FASTING)
        # ============================================================
        glucose = self.raw_data['glucose'][[
            'SEQN',
            'LBXGLU'     # Fasting glucose (mg/dL)
        ]].copy()
        print(f"  ✓ Glucose: {len(glucose)} rows")
        
        # ============================================================
        # CHOLESTEROL (TOTAL ONLY)
        # ============================================================
        chol = self.raw_data['cholesterol'][[
            'SEQN',
            'LBXTC'      # Total cholesterol (mg/dL)
        ]].copy()
        print(f"  ✓ Cholesterol: {len(chol)} rows (total cholesterol only)")
        
        # ============================================================
        # BIOCHEMISTRY
        # ============================================================
        biochem = self.raw_data['biochemistry'][[
            'SEQN',
            'LBXSCR',    # Creatinine (mg/dL)
            'LBXSGL',    # Glucose, refrigerated serum (mg/dL) - alternative
            'LBXSBU',    # Blood Urea Nitrogen (mg/dL)
            'LBXSTP',    # Total protein (g/dL)
            'LBXSUA'     # Uric acid (mg/dL)
        ]].copy()
        print(f"  ✓ Biochemistry: {len(biochem)} rows")
        
        # ============================================================
        # DIETARY (DAY 1 TOTALS)
        # ============================================================
        diet = self.raw_data['dietary'][[
            'SEQN',
            'DR1TKCAL',  # Total calories (kcal)
            'DR1TPROT',  # Protein (g)
            'DR1TCARB',  # Carbohydrate (g)
            'DR1TSUGR',  # Total sugars (g)
            'DR1TFIBE',  # Dietary fiber (g)
            'DR1TTFAT',  # Total fat (g)
            'DR1TSFAT',  # Saturated fatty acids (g)
            'DR1TSODI',  # Sodium (mg)
            'DR1TPOTA',  # Potassium (mg)
            'DR1TCHOL',  # Cholesterol (mg)
            'DR1TALCO'   # Alcohol (g)
        ]].copy()
        print(f"  ✓ Dietary: {len(diet)} rows")
        
        # ============================================================
        # PHYSICAL ACTIVITY - 2021-2023 USES DIFFERENT QUESTIONS
        # ============================================================
        # New coding:
        # PAD790Q/U = Vigorous recreational activities (times per week)
        # PAD810Q/U = Moderate recreational activities (times per week)
        # PAD800 = Minutes vigorous-intensity activity
        # PAD820 = Minutes moderate-intensity activity
        # PAD680 = Minutes sedentary activity
        
        activity = self.raw_data['physical_activity'][[
            'SEQN',
            'PAD790Q',   # Vigorous recreational activities (times/week)
            'PAD800',    # Minutes vigorous-intensity activity
            'PAD810Q',   # Moderate recreational activities (times/week)
            'PAD820',    # Minutes moderate-intensity activity
            'PAD680'     # Minutes sedentary activity (avg per day)
        ]].copy()
        print(f"  ✓ Physical activity: {len(activity)} rows")
        
        # ============================================================
        # DIABETES QUESTIONNAIRE
        # ============================================================
        diabetes = self.raw_data['diabetes'][[
            'SEQN',
            'DIQ010',    # Doctor told you have diabetes (1=Yes, 2=No, 3=Borderline)
            'DIQ160',    # Ever told you have prediabetes
            'DIQ050'     # Taking insulin now
        ]].copy()
        print(f"  ✓ Diabetes: {len(diabetes)} rows")
        
        # ============================================================
        # MEDICAL CONDITIONS
        # ============================================================
        medical = self.raw_data['medical_conditions'][[
            'SEQN',
            'MCQ160B',   # Ever told had congestive heart failure
            'MCQ160C',   # Ever told had coronary heart disease
            'MCQ160D',   # Ever told had angina/angina pectoris
            'MCQ160E',   # Ever told had heart attack
            'MCQ160F',   # Ever told had stroke
            'MCQ160L',   # Ever told had liver condition
            'MCQ160M',   # Ever told had thyroid problem
            'MCQ160P'    # Ever told had COPD/emphysema/chronic bronchitis
        ]].copy()
        print(f"  ✓ Medical conditions: {len(medical)} rows")
        
        # ============================================================
        # BLOOD PRESSURE QUESTIONNAIRE
        # ============================================================
        bp_quest = self.raw_data['bp_questionnaire'][[
            'SEQN',
            'BPQ020',    # Ever told you had high blood pressure
            'BPQ080'     # Doctor told you - high cholesterol level
        ]].copy()
        print(f"  ✓ BP questionnaire: {len(bp_quest)} rows")
        
        return {
            'demo': demo,
            'body': body,
            'bp': bp,
            'glucose': glucose,
            'cholesterol': chol,
            'biochem': biochem,
            'diet': diet,
            'activity': activity,
            'diabetes': diabetes,
            'medical': medical,
            'bp_quest': bp_quest
        }
    
    def merge_datasets(self, selected_data):
        """Merge all datasets on SEQN (participant ID)"""
        
        print("\nMerging datasets on SEQN...")
        
        # Start with demographics (has all participants)
        merged = selected_data['demo']
        print(f"  Starting with demographics: {len(merged)} rows")
        
        # Sequential left joins
        merge_order = ['body', 'bp', 'glucose', 'cholesterol', 'biochem', 
                       'diet', 'activity', 'diabetes', 'medical', 'bp_quest']
        
        for name in merge_order:
            before_cols = merged.shape[1]
            merged = merged.merge(selected_data[name], on='SEQN', how='left')
            added_cols = merged.shape[1] - before_cols
            print(f"  + {name}: {len(merged)} rows ({added_cols} new columns)")
        
        print(f"\n✓ Final merged dataset: {merged.shape[0]} rows × {merged.shape[1]} columns")
        return merged
    
    def clean_data(self, df):
        """Handle missing values and recode variables"""
        
        print("\nCleaning data...")
        
        # ============================================================
        # HANDLE MISSING VALUES
        # ============================================================
        # NHANES uses specific codes for missing data
        missing_codes = [7, 9, 77, 99, 777, 999, 7777, 9999]
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].replace(missing_codes, np.nan)
        
        print("  ✓ Replaced missing value codes with NaN")
        
        # ============================================================
        # RECODE BINARY VARIABLES
        # ============================================================
        # Map: 1=Yes→1, 2=No→0, 3=Borderline→1 (for diabetes)
        
        binary_mappings = {
            'DIQ010': ('has_diabetes', {1: 1, 2: 0, 3: 1}),  # 3=Borderline treated as Yes
            'DIQ160': ('has_prediabetes', {1: 1, 2: 0}),
            'DIQ050': ('uses_insulin', {1: 1, 2: 0}),
            'MCQ160B': ('has_heart_failure', {1: 1, 2: 0}),
            'MCQ160C': ('has_coronary_disease', {1: 1, 2: 0}),
            'MCQ160D': ('has_angina', {1: 1, 2: 0}),
            'MCQ160E': ('has_heart_attack', {1: 1, 2: 0}),
            'MCQ160F': ('has_stroke', {1: 1, 2: 0}),
            'MCQ160L': ('has_liver_condition', {1: 1, 2: 0}),
            'MCQ160M': ('has_thyroid_problem', {1: 1, 2: 0}),
            'MCQ160P': ('has_copd', {1: 1, 2: 0}),
            'BPQ020': ('has_hypertension', {1: 1, 2: 0}),
            'BPQ080': ('has_high_cholesterol_dx', {1: 1, 2: 0})
        }
        
        for old_col, (new_col, mapping) in binary_mappings.items():
            if old_col in df.columns:
                df[new_col] = df[old_col].replace(mapping)
                df = df.drop(columns=[old_col])
        
        print("  ✓ Recoded binary disease variables")
        
        # ============================================================
        # RENAME COLUMNS FOR CLARITY
        # ============================================================
        rename_map = {
            'SEQN': 'participant_id',
            'RIDAGEYR': 'age',
            'RIAGENDR': 'gender',
            'RIDRETH3': 'race',
            'DMDEDUC2': 'education',
            'INDFMPIR': 'income_poverty_ratio',
            'BMXBMI': 'bmi',
            'BMXWT': 'weight_kg',
            'BMXHT': 'height_cm',
            'BMXWAIST': 'waist_cm',
            'BMXHIP': 'hip_cm',
            'BPXOSY1': 'systolic_bp_1',
            'BPXODI1': 'diastolic_bp_1',
            'BPXOSY2': 'systolic_bp_2',
            'BPXODI2': 'diastolic_bp_2',
            'LBXGLU': 'glucose_fasting',
            'LBXSGL': 'glucose_serum',
            'LBXTC': 'total_cholesterol',
            'LBXSCR': 'creatinine',
            'LBXSBU': 'blood_urea_nitrogen',
            'LBXSTP': 'total_protein',
            'LBXSUA': 'uric_acid',
            'DR1TKCAL': 'calories',
            'DR1TPROT': 'protein_g',
            'DR1TCARB': 'carbs_g',
            'DR1TSUGR': 'sugar_g',
            'DR1TFIBE': 'fiber_g',
            'DR1TTFAT': 'total_fat_g',
            'DR1TSFAT': 'saturated_fat_g',
            'DR1TSODI': 'sodium_mg',
            'DR1TPOTA': 'potassium_mg',
            'DR1TCHOL': 'dietary_cholesterol_mg',
            'DR1TALCO': 'alcohol_g',
            'PAD790Q': 'vigorous_freq_per_week',
            'PAD800': 'vigorous_mins_per_day',
            'PAD810Q': 'moderate_freq_per_week',
            'PAD820': 'moderate_mins_per_day',
            'PAD680': 'sedentary_mins_per_day'
        }
        
        rename_map_filtered = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=rename_map_filtered)
        print("  ✓ Renamed columns")
        
        # ============================================================
        # CONVERT GENDER
        # ============================================================
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({1: 'Male', 2: 'Female'})
        
        # ============================================================
        # CREATE DERIVED VARIABLES
        # ============================================================
        
        # Average blood pressure (use mean of 2 readings)
        if 'systolic_bp_1' in df.columns and 'systolic_bp_2' in df.columns:
            df['systolic_bp'] = df[['systolic_bp_1', 'systolic_bp_2']].mean(axis=1)
            df['diastolic_bp'] = df[['diastolic_bp_1', 'diastolic_bp_2']].mean(axis=1)
            print("  ✓ Averaged blood pressure readings")
        
        # Unified glucose variable (prefer fasting)
        if 'glucose_fasting' in df.columns and 'glucose_serum' in df.columns:
            df['glucose'] = df['glucose_fasting'].fillna(df['glucose_serum'])
        elif 'glucose_fasting' in df.columns:
            df['glucose'] = df['glucose_fasting']
        elif 'glucose_serum' in df.columns:
            df['glucose'] = df['glucose_serum']
        
        # Total weekly activity minutes (CDC recommends 150 min/week)
        if 'vigorous_mins_per_day' in df.columns and 'moderate_mins_per_day' in df.columns:
            # Vigorous counts 2x as much as moderate
            df['weekly_activity_mins'] = (
                df['vigorous_mins_per_day'].fillna(0) * df['vigorous_freq_per_week'].fillna(0) +
                df['moderate_mins_per_day'].fillna(0) * df['moderate_freq_per_week'].fillna(0)
            )
            df['meets_activity_guidelines'] = (df['weekly_activity_mins'] >= 150).astype(int)
            print("  ✓ Calculated weekly activity metrics")
        
        # Waist-to-hip ratio (cardiovascular risk indicator)
        if 'waist_cm' in df.columns and 'hip_cm' in df.columns:
            df['waist_hip_ratio'] = df['waist_cm'] / df['hip_cm']
        
        # Cardiovascular disease composite (any CVD condition)
        cvd_cols = ['has_heart_failure', 'has_coronary_disease', 'has_angina', 
                    'has_heart_attack', 'has_stroke']
        if all(col in df.columns for col in cvd_cols):
            df['has_cvd'] = df[cvd_cols].max(axis=1)
            print("  ✓ Created CVD composite variable")
        
        # Metabolic syndrome risk indicators
        if all(col in df.columns for col in ['glucose', 'systolic_bp', 'bmi', 'waist_cm']):
            df['metabolic_risk_score'] = 0
            df.loc[df['glucose'] >= 100, 'metabolic_risk_score'] += 1
            df.loc[df['systolic_bp'] >= 130, 'metabolic_risk_score'] += 1
            df.loc[df['bmi'] >= 30, 'metabolic_risk_score'] += 1
            df.loc[(df['gender'] == 'Male') & (df['waist_cm'] >= 102), 'metabolic_risk_score'] += 1
            df.loc[(df['gender'] == 'Female') & (df['waist_cm'] >= 88), 'metabolic_risk_score'] += 1
            print("  ✓ Calculated metabolic risk score")
        
        print(f"\n✓ Final cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # ============================================================
        # DATA QUALITY SUMMARY
        # ============================================================
        missing_summary = df.isnull().sum()
        missing_pct = (missing_summary / len(df) * 100).round(1)
        
        print("\nMissing data summary (top 10):")
        top_missing = pd.DataFrame({
            'Column': missing_summary.index,
            'Missing': missing_summary.values,
            'Percent': missing_pct.values
        }).sort_values('Missing', ascending=False).head(10)
        
        print(top_missing.to_string(index=False))
        
        return df
    
    def process(self):
        """Full processing pipeline"""
        print("="*60)
        print("NHANES DATA PROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Select variables
        selected = self.select_variables()
        
        # Step 2: Merge datasets
        merged = self.merge_datasets(selected)
        
        # Step 3: Clean data
        cleaned = self.clean_data(merged)
        
        # Step 4: Save to CSV
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'nhanes_2021_2023_integrated.csv'
        
        cleaned.to_csv(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")
        
        # ============================================================
        # DATASET STATISTICS
        # ============================================================
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        print(f"\nTotal participants: {len(cleaned):,}")
        print(f"Age range: {cleaned['age'].min():.0f} - {cleaned['age'].max():.0f} years")
        print(f"Mean age: {cleaned['age'].mean():.1f} ± {cleaned['age'].std():.1f} years")
        
        if 'gender' in cleaned.columns:
            print(f"\nGender distribution:")
            print(cleaned['gender'].value_counts())
        
        print(f"\nDisease prevalence:")
        disease_cols = [
            ('Diabetes', 'has_diabetes'),
            ('Prediabetes', 'has_prediabetes'),
            ('Hypertension', 'has_hypertension'),
            ('High Cholesterol (dx)', 'has_high_cholesterol_dx'),
            ('Any CVD', 'has_cvd'),
            ('Heart Attack', 'has_heart_attack'),
            ('Stroke', 'has_stroke'),
            ('Heart Failure', 'has_heart_failure')
        ]
        
        for display_name, col_name in disease_cols:
            if col_name in cleaned.columns:
                count = cleaned[col_name].sum()
                total = cleaned[col_name].notna().sum()
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {display_name:25s}: {count:5.0f} / {total:5.0f} ({pct:5.1f}%)")
        
        # Clinical markers summary
        print(f"\nClinical markers (mean ± std):")
        clinical_cols = [
            ('Glucose (mg/dL)', 'glucose'),
            ('Total Cholesterol (mg/dL)', 'total_cholesterol'),
            ('Systolic BP (mmHg)', 'systolic_bp'),
            ('Diastolic BP (mmHg)', 'diastolic_bp'),
            ('BMI (kg/m²)', 'bmi'),
            ('Waist-Hip Ratio', 'waist_hip_ratio')
        ]
        
        for display_name, col_name in clinical_cols:
            if col_name in cleaned.columns:
                mean_val = cleaned[col_name].mean()
                std_val = cleaned[col_name].std()
                count = cleaned[col_name].notna().sum()
                print(f"  {display_name:30s}: {mean_val:6.1f} ± {std_val:5.1f} (n={count:,})")
        
        # Activity summary
        if 'weekly_activity_mins' in cleaned.columns:
            print(f"\nPhysical activity:")
            active_count = cleaned['meets_activity_guidelines'].sum()
            active_pct = (active_count / len(cleaned) * 100)
            print(f"  Meet CDC guidelines (≥150 min/week): {active_count:,} ({active_pct:.1f}%)")
            print(f"  Avg weekly activity: {cleaned['weekly_activity_mins'].mean():.0f} minutes")
        
        return cleaned


# Usage
if __name__ == "__main__":
    from data_loader_nhanes import NHANESDownloader
    
    print("Loading raw NHANES data...")
    downloader = NHANESDownloader()
    raw_data = downloader.load_all_data()
    
    print("\n" + "="*60 + "\n")
    
    # Process and merge
    processor = NHANESProcessor(raw_data)
    integrated_data = processor.process()
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run profile_generator.py to create disease-specific profiles")
    print("2. Build recommendation engine")
    print("3. Integrate with your MIMIC-based chatbot")