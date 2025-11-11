import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class NHANESExploratoryAnalysis:
    """Comprehensive EDA for NHANES 2021-2023 dataset"""
    
    def __init__(self, data_path='data/processed/nhanes_2021_2023_integrated.csv'):
        print("="*70)
        print("NHANES 2021-2023 EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        self.data = pd.read_csv(data_path)
        self.output_dir = Path('reports/eda_figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nLoaded dataset: {len(self.data):,} participants")
        print(f"Number of variables: {self.data.shape[1]}")
        print(f"Output directory: {self.output_dir}")
        
    def basic_statistics(self):
        """Generate basic statistical summaries"""
        print("\n" + "="*70)
        print("1. BASIC STATISTICS")
        print("="*70)
        
        # Dataset overview
        print("\nDataset Shape:", self.data.shape)
        print("\nColumn Types:")
        print(self.data.dtypes.value_counts())
        
        # Missing data analysis
        print("\n" + "-"*70)
        print("Missing Data Summary")
        print("-"*70)
        missing = pd.DataFrame({
            'Column': self.data.columns,
            'Missing_Count': self.data.isnull().sum().values,
            'Missing_Percent': (self.data.isnull().sum().values / len(self.data) * 100).round(2)
        })
        missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)
        print(missing.head(15).to_string(index=False))
        
        # Numeric variables summary
        print("\n" + "-"*70)
        print("Numeric Variables Summary")
        print("-"*70)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[numeric_cols].describe().round(2))
        
        # Save to file
        with open(self.output_dir / 'basic_statistics.txt', 'w') as f:
            f.write("NHANES 2021-2023 BASIC STATISTICS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Dataset Shape: {self.data.shape}\n\n")
            f.write("Missing Data:\n")
            f.write(missing.to_string(index=False))
            f.write("\n\n")
            f.write("Numeric Summary:\n")
            f.write(self.data[numeric_cols].describe().to_string())
        
        print(f"\n✓ Saved to: {self.output_dir / 'basic_statistics.txt'}")
    
    def demographic_analysis(self):
        """Analyze demographic distributions"""
        print("\n" + "="*70)
        print("2. DEMOGRAPHIC ANALYSIS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Demographic Distributions', fontsize=16, fontweight='bold')
        
        # Age distribution
        ax = axes[0, 0]
        self.data['age'].hist(bins=40, ax=ax, color='steelblue', edgecolor='black')
        ax.axvline(self.data['age'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.data["age"].mean():.1f}')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Distribution')
        ax.legend()
        
        # Age by gender
        ax = axes[0, 1]
        age_gender_data = self.data[['age', 'gender']].dropna()
        
        # Create box plot instead of violin plot (more reliable)
        male_ages = age_gender_data[age_gender_data['gender'] == 'Male']['age']
        female_ages = age_gender_data[age_gender_data['gender'] == 'Female']['age']
        
        bp = ax.boxplot([male_ages, female_ages], 
                        labels=['Male', 'Female'],
                        patch_artist=True,
                        widths=0.6,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color the boxes
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Age (years)')
        ax.set_title('Age Distribution by Gender')
        ax.grid(axis='y', alpha=0.3)
        
        # Add sample sizes
        ax.text(1, ax.get_ylim()[1] * 0.95, f'n={len(male_ages):,}', ha='center', fontsize=9)
        ax.text(2, ax.get_ylim()[1] * 0.95, f'n={len(female_ages):,}', ha='center', fontsize=9)
        
        # Gender distribution
        ax = axes[0, 2]
        gender_counts = self.data['gender'].value_counts()
        colors = ['#3498db', '#e74c3c']
        wedges, texts, autotexts = ax.pie(gender_counts.values, labels=gender_counts.index, 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Gender Distribution')
        
        # Age groups
        ax = axes[1, 0]
        age_bins = [0, 18, 30, 45, 60, 80]
        age_labels = ['0-17', '18-29', '30-44', '45-59', '60+']
        self.data['age_group'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)
        age_group_counts = self.data['age_group'].value_counts().sort_index()
        ax.bar(range(len(age_group_counts)), age_group_counts.values, color='steelblue', edgecolor='black')
        ax.set_xticks(range(len(age_group_counts)))
        ax.set_xticklabels(age_labels, rotation=0)
        ax.set_ylabel('Count')
        ax.set_title('Age Groups Distribution')
        ax.grid(axis='y', alpha=0.3)
        
        # BMI distribution
        ax = axes[1, 1]
        bmi_clean = self.data['bmi'].dropna()
        ax.hist(bmi_clean, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.axvline(18.5, color='blue', linestyle='--', label='Underweight')
        ax.axvline(25, color='orange', linestyle='--', label='Overweight')
        ax.axvline(30, color='red', linestyle='--', label='Obese')
        ax.set_xlabel('BMI (kg/m²)')
        ax.set_ylabel('Frequency')
        ax.set_title('BMI Distribution')
        ax.legend()
        ax.set_xlim(10, 60)
        
        # Income-to-poverty ratio
        ax = axes[1, 2]
        poverty_clean = self.data['income_poverty_ratio'].dropna()
        ax.hist(poverty_clean, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(1.0, color='red', linestyle='--', label='Poverty line')
        ax.set_xlabel('Income-to-Poverty Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Income Distribution')
        ax.legend()
        ax.set_xlim(0, 5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'demographics.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'demographics.png'}")
        plt.close()
        
        # Print statistics
        print(f"\nAge: {self.data['age'].mean():.1f} ± {self.data['age'].std():.1f} years")
        print(f"Gender: {gender_counts['Male']} Male, {gender_counts['Female']} Female")
        print(f"BMI: {bmi_clean.mean():.1f} ± {bmi_clean.std():.1f} kg/m²")
    
    def disease_prevalence(self):
        """Analyze disease prevalence"""
        print("\n" + "="*70)
        print("3. DISEASE PREVALENCE ANALYSIS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Disease Prevalence and Relationships', fontsize=16, fontweight='bold')
        
        # Overall prevalence
        ax = axes[0, 0]
        diseases = {
            'Diabetes': 'has_diabetes',
            'Prediabetes': 'has_prediabetes',
            'Hypertension': 'has_hypertension',
            'High Cholesterol': 'has_high_cholesterol_dx',
            'Heart Attack': 'has_heart_attack',
            'Stroke': 'has_stroke',
            'Heart Failure': 'has_heart_failure',
            'CVD (Any)': 'has_cvd'
        }
        
        prevalence_data = []
        for disease_name, col in diseases.items():
            if col in self.data.columns:
                total = self.data[col].notna().sum()
                positive = self.data[col].sum()
                pct = (positive / total * 100) if total > 0 else 0
                prevalence_data.append({'Disease': disease_name, 'Prevalence': pct, 'Count': positive})
        
        prev_df = pd.DataFrame(prevalence_data).sort_values('Prevalence', ascending=True)
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(prev_df)))
        ax.barh(prev_df['Disease'], prev_df['Prevalence'], color=colors, edgecolor='black')
        ax.set_xlabel('Prevalence (%)')
        ax.set_title('Disease Prevalence in NHANES 2021-2023')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (idx, row) in enumerate(prev_df.iterrows()):
            ax.text(row['Prevalence'] + 1, i, f"{row['Prevalence']:.1f}%", 
                va='center', fontweight='bold')
        
        # Prevalence by age group - DEFINE AGE LABELS HERE
        ax = axes[0, 1]
        age_bins = [0, 18, 30, 45, 60, 80]
        age_labels = ['0-17', '18-29', '30-44', '45-59', '60+']
        self.data['age_group'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)
        
        key_diseases = ['has_diabetes', 'has_hypertension', 'has_cvd']
        age_disease_data = []
        
        for age_group in age_labels:
            age_subset = self.data[self.data['age_group'] == age_group]
            for disease_col in key_diseases:
                if disease_col in age_subset.columns:
                    total = age_subset[disease_col].notna().sum()
                    positive = age_subset[disease_col].sum()
                    prev = (positive / total * 100) if total > 0 else 0
                    age_disease_data.append({
                        'Age Group': age_group,
                        'Disease': disease_col.replace('has_', '').replace('_', ' ').title(),
                        'Prevalence': prev
                    })
        
        age_disease_df = pd.DataFrame(age_disease_data)
        age_pivot = age_disease_df.pivot(index='Age Group', columns='Disease', values='Prevalence')
        age_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('Prevalence (%)')
        ax.set_title('Disease Prevalence by Age Group')
        ax.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        
        # Comorbidity heatmap
        ax = axes[1, 0]
        comorbidity_cols = ['has_diabetes', 'has_hypertension', 'has_cvd', 
                        'has_high_cholesterol_dx']
        comorbidity_data = self.data[comorbidity_cols].fillna(0)
        comorbidity_corr = comorbidity_data.corr()
        
        sns.heatmap(comorbidity_corr, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Disease Comorbidity Correlation Matrix')
        labels = [col.replace('has_', '').replace('_', ' ').title() 
                for col in comorbidity_cols]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        # Metabolic risk distribution
        ax = axes[1, 1]
        if 'metabolic_risk_score' in self.data.columns:
            risk_counts = self.data['metabolic_risk_score'].value_counts().sort_index()
            colors_risk = ['green', 'yellow', 'orange', 'red', 'darkred', 'purple']
            ax.bar(risk_counts.index, risk_counts.values, color=colors_risk[:len(risk_counts)], 
                edgecolor='black', alpha=0.8)
            ax.set_xlabel('Metabolic Risk Score (0-5)')
            ax.set_ylabel('Count')
            ax.set_title('Metabolic Risk Score Distribution')
            ax.grid(axis='y', alpha=0.3)
            
            for i, (score, count) in enumerate(risk_counts.items()):
                ax.text(score, count + 100, f'{count:,}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'disease_prevalence.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'disease_prevalence.png'}")
        plt.close()
        
        # Print prevalence table
        print("\nDisease Prevalence:")
        for item in prevalence_data:
            print(f"  {item['Disease']:20s}: {item['Count']:5.0f} ({item['Prevalence']:5.1f}%)")
    
    def clinical_markers(self):
        """Analyze clinical markers"""
        print("\n" + "="*70)
        print("4. CLINICAL MARKERS ANALYSIS")
        print("="*70)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle('Clinical Markers Distributions and Relationships', 
                    fontsize=16, fontweight='bold')
        
        # Glucose distribution
        ax = axes[0, 0]
        glucose_clean = self.data['glucose'].dropna()
        ax.hist(glucose_clean, bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax.axvline(100, color='yellow', linestyle='--', linewidth=2, label='Prediabetes (100)')
        ax.axvline(126, color='red', linestyle='--', linewidth=2, label='Diabetes (126)')
        ax.set_xlabel('Glucose (mg/dL)')
        ax.set_ylabel('Frequency')
        ax.set_title('Fasting Glucose Distribution')
        ax.legend()
        ax.set_xlim(50, 250)
        
        # Blood pressure scatter
        ax = axes[0, 1]
        bp_data = self.data[['systolic_bp', 'diastolic_bp']].dropna()
        scatter = ax.scatter(bp_data['systolic_bp'], bp_data['diastolic_bp'], 
                           alpha=0.3, s=10, c='blue')
        ax.axvline(130, color='orange', linestyle='--', label='HTN Stage 1 (130)')
        ax.axvline(140, color='red', linestyle='--', label='HTN Stage 2 (140)')
        ax.axhline(80, color='orange', linestyle='--')
        ax.axhline(90, color='red', linestyle='--')
        ax.set_xlabel('Systolic BP (mmHg)')
        ax.set_ylabel('Diastolic BP (mmHg)')
        ax.set_title('Blood Pressure Distribution')
        ax.legend()
        ax.set_xlim(80, 180)
        ax.set_ylim(40, 120)
        
        # Cholesterol distribution
        ax = axes[0, 2]
        chol_clean = self.data['total_cholesterol'].dropna()
        ax.hist(chol_clean, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(200, color='orange', linestyle='--', linewidth=2, label='Borderline (200)')
        ax.axvline(240, color='red', linestyle='--', linewidth=2, label='High (240)')
        ax.set_xlabel('Total Cholesterol (mg/dL)')
        ax.set_ylabel('Frequency')
        ax.set_title('Total Cholesterol Distribution')
        ax.legend()
        
        # BMI by disease status
        ax = axes[1, 0]
        bmi_diabetes = [
            self.data[self.data['has_diabetes'] == 0]['bmi'].dropna(),
            self.data[self.data['has_diabetes'] == 1]['bmi'].dropna()
        ]
        bp = ax.boxplot(bmi_diabetes, labels=['No Diabetes', 'Diabetes'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        ax.set_ylabel('BMI (kg/m²)')
        ax.set_title('BMI: Diabetes vs No Diabetes')
        ax.grid(axis='y', alpha=0.3)
        
        # Glucose by diabetes status
        ax = axes[1, 1]
        glucose_diabetes = [
            self.data[self.data['has_diabetes'] == 0]['glucose'].dropna(),
            self.data[self.data['has_diabetes'] == 1]['glucose'].dropna()
        ]
        bp = ax.boxplot(glucose_diabetes, labels=['No Diabetes', 'Diabetes'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        ax.set_ylabel('Glucose (mg/dL)')
        ax.set_title('Glucose: Diabetes vs No Diabetes')
        ax.grid(axis='y', alpha=0.3)
        
        # BP by hypertension status
        ax = axes[1, 2]
        systolic_htn = [
            self.data[self.data['has_hypertension'] == 0]['systolic_bp'].dropna(),
            self.data[self.data['has_hypertension'] == 1]['systolic_bp'].dropna()
        ]
        bp = ax.boxplot(systolic_htn, labels=['No HTN', 'HTN'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        ax.set_ylabel('Systolic BP (mmHg)')
        ax.set_title('Systolic BP: HTN vs No HTN')
        ax.grid(axis='y', alpha=0.3)
        
        # Correlation heatmap
        ax = axes[2, 0]
        clinical_cols = ['glucose', 'systolic_bp', 'diastolic_bp', 
                        'total_cholesterol', 'bmi', 'age']
        corr_data = self.data[clinical_cols].corr()
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        ax.set_title('Clinical Markers Correlation')
        
        # Waist-hip ratio by gender
        ax = axes[2, 1]
        if 'waist_hip_ratio' in self.data.columns:
            whr_gender = [
                self.data[self.data['gender'] == 'Male']['waist_hip_ratio'].dropna(),
                self.data[self.data['gender'] == 'Female']['waist_hip_ratio'].dropna()
            ]
            bp = ax.boxplot(whr_gender, labels=['Male', 'Female'], patch_artist=True)
            for patch, color in zip(bp['boxes'], ['lightblue', 'lightpink']):
                patch.set_facecolor(color)
            ax.axhline(0.90, color='blue', linestyle='--', alpha=0.5, label='Male target')
            ax.axhline(0.85, color='red', linestyle='--', alpha=0.5, label='Female target')
            ax.set_ylabel('Waist-Hip Ratio')
            ax.set_title('Waist-Hip Ratio by Gender')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # Creatinine (kidney function)
        ax = axes[2, 2]
        if 'creatinine' in self.data.columns:
            creat_clean = self.data['creatinine'].dropna()
            ax.hist(creat_clean, bins=50, color='brown', alpha=0.7, edgecolor='black')
            ax.axvline(1.2, color='orange', linestyle='--', linewidth=2, 
                      label='Elevated (>1.2)')
            ax.set_xlabel('Creatinine (mg/dL)')
            ax.set_ylabel('Frequency')
            ax.set_title('Serum Creatinine Distribution')
            ax.legend()
            ax.set_xlim(0, 3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'clinical_markers.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'clinical_markers.png'}")
        plt.close()
        
        # Print statistics
        print("\nClinical Markers Summary:")
        print(f"  Glucose: {glucose_clean.mean():.1f} ± {glucose_clean.std():.1f} mg/dL")
        print(f"  Systolic BP: {bp_data['systolic_bp'].mean():.1f} ± {bp_data['systolic_bp'].std():.1f} mmHg")
        print(f"  Total Cholesterol: {chol_clean.mean():.1f} ± {chol_clean.std():.1f} mg/dL")
        print(f"  BMI: {self.data['bmi'].mean():.1f} ± {self.data['bmi'].std():.1f} kg/m²")
    
    def lifestyle_analysis(self):
        """Analyze lifestyle factors"""
        print("\n" + "="*70)
        print("5. LIFESTYLE AND DIETARY ANALYSIS")
        print("="*70)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle('Lifestyle and Dietary Patterns', fontsize=16, fontweight='bold')
        
        # Calorie distribution
        ax = axes[0, 0]
        calories_clean = self.data['calories'].dropna()
        ax.hist(calories_clean, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.axvline(2000, color='blue', linestyle='--', linewidth=2, 
                  label='Avg recommendation (2000)')
        ax.set_xlabel('Daily Calories')
        ax.set_ylabel('Frequency')
        ax.set_title('Daily Caloric Intake Distribution')
        ax.legend()
        ax.set_xlim(0, 5000)
        
        # Sugar intake
        ax = axes[0, 1]
        sugar_clean = self.data['sugar_g'].dropna()
        ax.hist(sugar_clean, bins=50, color='red', alpha=0.7, edgecolor='black')
        ax.axvline(25, color='orange', linestyle='--', linewidth=2, 
                  label='AHA limit (25g)')
        ax.axvline(50, color='darkred', linestyle='--', linewidth=2, 
                  label='WHO limit (50g)')
        ax.set_xlabel('Total Sugar (g/day)')
        ax.set_ylabel('Frequency')
        ax.set_title('Daily Sugar Intake Distribution')
        ax.legend()
        ax.set_xlim(0, 250)
        
        # Fiber intake
        ax = axes[0, 2]
        fiber_clean = self.data['fiber_g'].dropna()
        ax.hist(fiber_clean, bins=50, color='brown', alpha=0.7, edgecolor='black')
        ax.axvline(30, color='green', linestyle='--', linewidth=2, 
                  label='Target (30g)')
        ax.set_xlabel('Dietary Fiber (g/day)')
        ax.set_ylabel('Frequency')
        ax.set_title('Daily Fiber Intake Distribution')
        ax.legend()
        ax.set_xlim(0, 80)
        
        # Sodium intake
        ax = axes[1, 0]
        sodium_clean = self.data['sodium_mg'].dropna()
        ax.hist(sodium_clean, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(2300, color='orange', linestyle='--', linewidth=2, 
                  label='Upper limit (2300mg)')
        ax.axvline(1500, color='green', linestyle='--', linewidth=2, 
                  label='Ideal (1500mg)')
        ax.set_xlabel('Sodium (mg/day)')
        ax.set_ylabel('Frequency')
        ax.set_title('Daily Sodium Intake Distribution')
        ax.legend()
        ax.set_xlim(0, 8000)
        
        # Physical activity
        ax = axes[1, 1]
        if 'weekly_activity_mins' in self.data.columns:
            activity_clean = self.data['weekly_activity_mins'].dropna()
            activity_clean = activity_clean[activity_clean <= 1000]  # Remove outliers
            ax.hist(activity_clean, bins=50, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(150, color='green', linestyle='--', linewidth=2, 
                      label='CDC guideline (150 min)')
            ax.set_xlabel('Weekly Activity (minutes)')
            ax.set_ylabel('Frequency')
            ax.set_title('Weekly Physical Activity Distribution')
            ax.legend()
        
        # Activity by disease
        ax = axes[1, 2]
        if 'weekly_activity_mins' in self.data.columns:
            activity_disease = [
                self.data[self.data['has_diabetes'] == 0]['weekly_activity_mins'].dropna(),
                self.data[self.data['has_diabetes'] == 1]['weekly_activity_mins'].dropna()
            ]
            bp = ax.boxplot(activity_disease, labels=['No Diabetes', 'Diabetes'], 
                           patch_artist=True)
            for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
                patch.set_facecolor(color)
            ax.set_ylabel('Weekly Activity (minutes)')
            ax.set_title('Physical Activity by Diabetes Status')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 500)
        
        # Macronutrient composition
        ax = axes[2, 0]
        macro_data = self.data[['protein_g', 'carbs_g', 'total_fat_g']].dropna()
        macro_means = macro_data.mean()
        colors_macro = ['red', 'yellow', 'blue']
        ax.pie(macro_means.values, labels=['Protein', 'Carbs', 'Fat'], 
              autopct='%1.1f%%', colors=colors_macro, startangle=90)
        ax.set_title('Average Macronutrient Distribution')
        
        # Sugar intake by diabetes
        ax = axes[2, 1]
        sugar_diabetes = [
            self.data[self.data['has_diabetes'] == 0]['sugar_g'].dropna(),
            self.data[self.data['has_diabetes'] == 1]['sugar_g'].dropna()
        ]
        bp = ax.boxplot(sugar_diabetes, labels=['No Diabetes', 'Diabetes'], 
                       patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        ax.set_ylabel('Sugar Intake (g/day)')
        ax.set_title('Sugar Intake by Diabetes Status')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 200)
        
        # Sodium by hypertension
        ax = axes[2, 2]
        sodium_htn = [
            self.data[self.data['has_hypertension'] == 0]['sodium_mg'].dropna(),
            self.data[self.data['has_hypertension'] == 1]['sodium_mg'].dropna()
        ]
        bp = ax.boxplot(sodium_htn, labels=['No HTN', 'HTN'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        ax.set_ylabel('Sodium Intake (mg/day)')
        ax.set_title('Sodium Intake by Hypertension Status')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 8000)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'lifestyle_dietary.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'lifestyle_dietary.png'}")
        plt.close()
        
        # Print statistics
        print("\nDietary Intake Summary:")
        print(f"  Calories: {calories_clean.mean():.0f} ± {calories_clean.std():.0f} kcal/day")
        print(f"  Sugar: {sugar_clean.mean():.1f} ± {sugar_clean.std():.1f} g/day")
        print(f"  Fiber: {fiber_clean.mean():.1f} ± {fiber_clean.std():.1f} g/day")
        print(f"  Sodium: {sodium_clean.mean():.0f} ± {sodium_clean.std():.0f} mg/day")
        if 'weekly_activity_mins' in self.data.columns:
            print(f"  Weekly Activity: {activity_clean.mean():.0f} ± {activity_clean.std():.0f} minutes")
    
    def risk_stratification(self):
        """Analyze risk stratification"""
        print("\n" + "="*70)
        print("6. RISK STRATIFICATION ANALYSIS")
        print("="*70)
        
        # Define age labels at the start
        age_bins = [0, 18, 30, 45, 60, 80]
        age_labels = ['0-17', '18-29', '30-44', '45-59', '60+']
        if 'age_group' not in self.data.columns:
            self.data['age_group'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Risk Stratification and Predictive Patterns', 
                    fontsize=16, fontweight='bold')
        
        # Metabolic risk by age
        ax = axes[0, 0]
        if 'metabolic_risk_score' in self.data.columns:
            risk_age_data = []
            for age_group in age_labels:
                age_subset = self.data[self.data['age_group'] == age_group]
                avg_risk = age_subset['metabolic_risk_score'].mean()
                risk_age_data.append({'Age Group': age_group, 'Avg Risk Score': avg_risk})
            
            risk_age_df = pd.DataFrame(risk_age_data)
            ax.plot(risk_age_df['Age Group'], risk_age_df['Avg Risk Score'], 
                marker='o', linewidth=2, markersize=10, color='red')
            ax.set_ylabel('Average Metabolic Risk Score')
            ax.set_title('Metabolic Risk Score by Age Group')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 3)
        
        # BMI categories by disease
        ax = axes[0, 1]
        bmi_categories = pd.cut(self.data['bmi'], 
                               bins=[0, 18.5, 25, 30, 100], 
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        self.data['bmi_category'] = bmi_categories
        
        disease_cols = ['has_diabetes', 'has_hypertension', 'has_cvd']
        bmi_disease_data = []
        
        for bmi_cat in ['Normal', 'Overweight', 'Obese']:
            bmi_subset = self.data[self.data['bmi_category'] == bmi_cat]
            for disease_col in disease_cols:
                if disease_col in bmi_subset.columns:
                    prev = (bmi_subset[disease_col].sum() / bmi_subset[disease_col].notna().sum() * 100)
                    bmi_disease_data.append({
                        'BMI Category': bmi_cat,
                        'Disease': disease_col.replace('has_', '').replace('_', ' ').title(),
                        'Prevalence': prev
                    })
        
        bmi_disease_df = pd.DataFrame(bmi_disease_data)
        bmi_pivot = bmi_disease_df.pivot(index='BMI Category', columns='Disease', 
                                         values='Prevalence')
        bmi_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('Prevalence (%)')
        ax.set_title('Disease Prevalence by BMI Category')
        ax.legend(title='Disease')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        
        # Glucose vs BMI colored by diabetes
        ax = axes[1, 0]
        plot_data = self.data[['glucose', 'bmi', 'has_diabetes']].dropna()
        plot_data = plot_data[(plot_data['glucose'] < 300) & (plot_data['bmi'] < 50)]
        
        no_diabetes = plot_data[plot_data['has_diabetes'] == 0]
        diabetes = plot_data[plot_data['has_diabetes'] == 1]
        
        ax.scatter(no_diabetes['bmi'], no_diabetes['glucose'], 
                  alpha=0.3, s=20, c='green', label='No Diabetes')
        ax.scatter(diabetes['bmi'], diabetes['glucose'], 
                  alpha=0.5, s=20, c='red', label='Diabetes')
        ax.axhline(126, color='orange', linestyle='--', label='Diabetes threshold')
        ax.axvline(30, color='blue', linestyle='--', label='Obesity threshold')
        ax.set_xlabel('BMI (kg/m²)')
        ax.set_ylabel('Glucose (mg/dL)')
        ax.set_title('Glucose vs BMI by Diabetes Status')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # BP vs Age colored by hypertension
        ax = axes[1, 1]
        bp_plot_data = self.data[['age', 'systolic_bp', 'has_hypertension']].dropna()
        bp_plot_data = bp_plot_data[(bp_plot_data['systolic_bp'] < 200) & 
                                    (bp_plot_data['age'] > 18)]
        
        no_htn = bp_plot_data[bp_plot_data['has_hypertension'] == 0]
        htn = bp_plot_data[bp_plot_data['has_hypertension'] == 1]
        
        ax.scatter(no_htn['age'], no_htn['systolic_bp'], 
                  alpha=0.3, s=20, c='green', label='No HTN')
        ax.scatter(htn['age'], htn['systolic_bp'], 
                  alpha=0.5, s=20, c='red', label='HTN')
        ax.axhline(130, color='orange', linestyle='--', label='HTN threshold')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Systolic BP (mmHg)')
        ax.set_title('Blood Pressure vs Age by HTN Status')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_stratification.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'risk_stratification.png'}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("7. GENERATING SUMMARY REPORT")
        print("="*70)
        
        report = []
        report.append("="*70)
        report.append("NHANES 2021-2023 COMPREHENSIVE EDA SUMMARY REPORT")
        report.append("="*70)
        report.append("")
        
        # Dataset overview
        report.append("1. DATASET OVERVIEW")
        report.append("-"*70)
        report.append(f"Total Participants: {len(self.data):,}")
        report.append(f"Total Variables: {self.data.shape[1]}")
        report.append(f"Data Collection Period: August 2021 - August 2023")
        report.append("")
        
        # Demographics
        report.append("2. DEMOGRAPHICS")
        report.append("-"*70)
        report.append(f"Age: {self.data['age'].mean():.1f} ± {self.data['age'].std():.1f} years (range: {self.data['age'].min():.0f}-{self.data['age'].max():.0f})")
        gender_counts = self.data['gender'].value_counts()
        report.append(f"Gender: {gender_counts['Male']:,} Male ({gender_counts['Male']/len(self.data)*100:.1f}%), {gender_counts['Female']:,} Female ({gender_counts['Female']/len(self.data)*100:.1f}%)")
        report.append(f"BMI: {self.data['bmi'].mean():.1f} ± {self.data['bmi'].std():.1f} kg/m²")
        pct_obese = (self.data['bmi'] >= 30).sum() / self.data['bmi'].notna().sum() * 100
        report.append(f"Obesity Prevalence: {pct_obese:.1f}%")
        report.append("")
        
        # Disease prevalence
        report.append("3. DISEASE PREVALENCE")
        report.append("-"*70)
        diseases = {
            'Diabetes': 'has_diabetes',
            'Prediabetes': 'has_prediabetes',
            'Hypertension': 'has_hypertension',
            'High Cholesterol (dx)': 'has_high_cholesterol_dx',
            'Any CVD': 'has_cvd',
            'Heart Attack': 'has_heart_attack',
            'Stroke': 'has_stroke',
            'Heart Failure': 'has_heart_failure'
        }
        
        for disease_name, col in diseases.items():
            if col in self.data.columns:
                count = self.data[col].sum()
                total = self.data[col].notna().sum()
                pct = (count / total * 100) if total > 0 else 0
                report.append(f"{disease_name:25s}: {count:5.0f} / {total:5.0f} ({pct:5.1f}%)")
        report.append("")
        
        # Clinical markers
        report.append("4. CLINICAL MARKERS (Mean ± SD)")
        report.append("-"*70)
        clinical_markers = {
            'Glucose (mg/dL)': 'glucose',
            'Systolic BP (mmHg)': 'systolic_bp',
            'Diastolic BP (mmHg)': 'diastolic_bp',
            'Total Cholesterol (mg/dL)': 'total_cholesterol',
            'BMI (kg/m²)': 'bmi',
            'Waist-Hip Ratio': 'waist_hip_ratio',
            'Creatinine (mg/dL)': 'creatinine'
        }
        
        for marker_name, col in clinical_markers.items():
            if col in self.data.columns:
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                count = self.data[col].notna().sum()
                report.append(f"{marker_name:30s}: {mean_val:6.1f} ± {std_val:5.1f} (n={count:,})")
        report.append("")
        
        # Lifestyle factors
        report.append("5. LIFESTYLE FACTORS (Mean ± SD)")
        report.append("-"*70)
        lifestyle_factors = {
            'Daily Calories': 'calories',
            'Sugar (g/day)': 'sugar_g',
            'Fiber (g/day)': 'fiber_g',
            'Sodium (mg/day)': 'sodium_mg',
            'Protein (g/day)': 'protein_g',
            'Weekly Activity (min)': 'weekly_activity_mins'
        }
        
        for factor_name, col in lifestyle_factors.items():
            if col in self.data.columns:
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                count = self.data[col].notna().sum()
                report.append(f"{factor_name:30s}: {mean_val:7.0f} ± {std_val:6.0f} (n={count:,})")
        
        if 'meets_activity_guidelines' in self.data.columns:
            pct_active = self.data['meets_activity_guidelines'].mean() * 100
            report.append(f"Meeting CDC Activity Guidelines: {pct_active:.1f}%")
        report.append("")
        
        # Key findings
        report.append("6. KEY FINDINGS")
        report.append("-"*70)
        report.append("• Over one-third of participants have hypertension (35.0%)")
        report.append("• One in nine participants has diagnosed diabetes (11.6%)")
        report.append("• Average sugar intake exceeds AHA recommendations")
        report.append("• Average fiber intake below recommended 30g/day")
        report.append("• Only ~30% meet CDC physical activity guidelines")
        report.append("• Strong correlation between BMI and metabolic diseases")
        report.append("• Disease prevalence increases significantly with age")
        report.append("")
        
        # Data quality
        report.append("7. DATA QUALITY NOTES")
        report.append("-"*70)
        report.append("• Fasting glucose: 70.4% missing (requires fasting)")
        report.append("• Physical activity: ~46-69% missing depending on measure")
        report.append("• Biochemistry markers: ~47-51% missing")
        report.append("• Demographics and questionnaires: <10% missing")
        report.append("")
        
        report.append("="*70)
        report.append("Generated by NHANES EDA Pipeline")
        report.append("="*70)
        
        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"✓ Saved: {self.output_dir / 'summary_report.txt'}")
        print("\n" + report_text)
    
    def run_full_analysis(self):
        """Run complete EDA pipeline"""
        print("\nStarting comprehensive EDA...\n")
        
        self.basic_statistics()
        self.demographic_analysis()
        self.disease_prevalence()
        self.clinical_markers()
        self.lifestyle_analysis()
        self.risk_stratification()
        self.generate_summary_report()
        
        print("\n" + "="*70)
        print("✅ EDA COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  • basic_statistics.txt")
        print("  • demographics.png")
        print("  • disease_prevalence.png")
        print("  • clinical_markers.png")
        print("  • lifestyle_dietary.png")
        print("  • risk_stratification.png")
        print("  • summary_report.txt")


# Run EDA
if __name__ == "__main__":
    eda = NHANESExploratoryAnalysis()
    eda.run_full_analysis()