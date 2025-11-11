import pandas as pd
import numpy as np
import json
from pathlib import Path

class DiseaseProfileGenerator:
    """Generate disease-specific population profiles from NHANES data"""
    
    def __init__(self, nhanes_data):
        self.data = nhanes_data
        
    def create_diabetes_profile(self):
        """Profile of people with diabetes"""
        print("\nGenerating Diabetes profile...")
        
        diabetic = self.data[self.data['has_diabetes'] == 1].copy()
        non_diabetic = self.data[self.data['has_diabetes'] == 0].copy()
        
        profile = {
            'disease': 'Diabetes',
            'description': 'Population with diagnosed diabetes mellitus',
            'sample_size': int(diabetic['has_diabetes'].sum()),
            
            'demographics': {
                'avg_age': float(diabetic['age'].mean()),
                'age_std': float(diabetic['age'].std()),
                'gender_distribution': diabetic['gender'].value_counts().to_dict(),
                'pct_over_45': float((diabetic['age'] >= 45).mean() * 100)
            },
            
            'clinical_markers': {
                'glucose': {
                    'mean': float(diabetic['glucose'].mean()),
                    'median': float(diabetic['glucose'].median()),
                    'std': float(diabetic['glucose'].std()),
                    'percentile_75': float(diabetic['glucose'].quantile(0.75)),
                    'percentile_90': float(diabetic['glucose'].quantile(0.90)),
                    'target': '<100 mg/dL (fasting)',
                    'prediabetes_threshold': '100-125 mg/dL',
                    'diabetes_threshold': '≥126 mg/dL'
                },
                'bmi': {
                    'mean': float(diabetic['bmi'].mean()),
                    'median': float(diabetic['bmi'].median()),
                    'pct_obese': float((diabetic['bmi'] >= 30).mean() * 100),
                    'target': '18.5-24.9 kg/m²'
                },
                'blood_pressure': {
                    'systolic_mean': float(diabetic['systolic_bp'].mean()),
                    'diastolic_mean': float(diabetic['diastolic_bp'].mean()),
                    'pct_hypertensive': float(diabetic['has_hypertension'].mean() * 100),
                    'target': '<130/80 mmHg for diabetics'
                },
                'cholesterol': {
                    'mean': float(diabetic['total_cholesterol'].mean()),
                    'target': '<200 mg/dL'
                }
            },
            
            'lifestyle_patterns': {
                'dietary': {
                    'avg_calories': float(diabetic['calories'].mean()),
                    'avg_sugar_g': float(diabetic['sugar_g'].mean()),
                    'avg_fiber_g': float(diabetic['fiber_g'].mean()),
                    'avg_carbs_g': float(diabetic['carbs_g'].mean()),
                    'avg_sodium_mg': float(diabetic['sodium_mg'].mean())
                },
                'activity': {
                    'avg_weekly_mins': float(diabetic['weekly_activity_mins'].mean()),
                    'pct_meeting_guidelines': float(diabetic['meets_activity_guidelines'].mean() * 100),
                    'avg_sedentary_mins': float(diabetic['sedentary_mins_per_day'].mean())
                }
            },
            
            'comparison_to_healthy': {
                'glucose_difference': float(diabetic['glucose'].mean() - non_diabetic['glucose'].mean()),
                'bmi_difference': float(diabetic['bmi'].mean() - non_diabetic['bmi'].mean()),
                'activity_difference': float(diabetic['weekly_activity_mins'].mean() - 
                                             non_diabetic['weekly_activity_mins'].mean())
            },
            
            'recommendations': {
                'glucose_control': {
                    'target': 'Fasting glucose <100 mg/dL',
                    'action': 'Monitor glucose regularly, limit added sugars'
                },
                'diet': {
                    'sugar_limit': '<25g added sugars per day',
                    'fiber_goal': '>30g fiber per day',
                    'carb_strategy': '45-60g carbs per meal, focus on complex carbs',
                    'sodium_limit': '<2,300mg sodium per day'
                },
                'activity': {
                    'goal': '150 minutes moderate activity per week',
                    'strength_training': '2 days per week',
                    'benefit': 'Improves insulin sensitivity and glucose control'
                },
                'weight': {
                    'target': 'BMI 18.5-24.9, or 5-10% weight loss if overweight',
                    'benefit': 'Even modest weight loss improves glycemic control'
                }
            }
        }
        
        print(f"  ✓ Diabetes: {profile['sample_size']} participants")
        return profile
    
    def create_hypertension_profile(self):
        """Profile of people with hypertension"""
        print("\nGenerating Hypertension profile...")
        
        hypertensive = self.data[self.data['has_hypertension'] == 1].copy()
        normotensive = self.data[self.data['has_hypertension'] == 0].copy()
        
        profile = {
            'disease': 'Hypertension',
            'description': 'Population with diagnosed high blood pressure',
            'sample_size': int(hypertensive['has_hypertension'].sum()),
            
            'demographics': {
                'avg_age': float(hypertensive['age'].mean()),
                'age_std': float(hypertensive['age'].std()),
                'gender_distribution': hypertensive['gender'].value_counts().to_dict()
            },
            
            'clinical_markers': {
                'blood_pressure': {
                    'systolic_mean': float(hypertensive['systolic_bp'].mean()),
                    'systolic_median': float(hypertensive['systolic_bp'].median()),
                    'diastolic_mean': float(hypertensive['diastolic_bp'].mean()),
                    'diastolic_median': float(hypertensive['diastolic_bp'].median()),
                    'percentile_75_systolic': float(hypertensive['systolic_bp'].quantile(0.75)),
                    'target': '<120/80 mmHg (normal), <130/80 mmHg (acceptable with HTN)'
                },
                'bmi': {
                    'mean': float(hypertensive['bmi'].mean()),
                    'pct_overweight': float((hypertensive['bmi'] >= 25).mean() * 100)
                },
                'waist_hip_ratio': {
                    'mean': float(hypertensive['waist_hip_ratio'].mean()),
                    'target_male': '<0.90',
                    'target_female': '<0.85'
                }
            },
            
            'lifestyle_patterns': {
                'dietary': {
                    'avg_sodium_mg': float(hypertensive['sodium_mg'].mean()),
                    'avg_potassium_mg': float(hypertensive['potassium_mg'].mean()),
                    'avg_calories': float(hypertensive['calories'].mean())
                },
                'activity': {
                    'avg_weekly_mins': float(hypertensive['weekly_activity_mins'].mean()),
                    'pct_meeting_guidelines': float(hypertensive['meets_activity_guidelines'].mean() * 100)
                }
            },
            
            'comparison_to_healthy': {
                'systolic_difference': float(hypertensive['systolic_bp'].mean() - 
                                            normotensive['systolic_bp'].mean()),
                'sodium_difference': float(hypertensive['sodium_mg'].mean() - 
                                          normotensive['sodium_mg'].mean()),
                'bmi_difference': float(hypertensive['bmi'].mean() - normotensive['bmi'].mean())
            },
            
            'recommendations': {
                'blood_pressure': {
                    'target': '<130/80 mmHg',
                    'monitor': 'Check BP regularly at home'
                },
                'diet': {
                    'sodium_limit': '<2,300mg per day (ideally <1,500mg)',
                    'potassium_goal': '>3,500mg per day (from food sources)',
                    'dash_diet': 'Follow DASH diet: fruits, vegetables, whole grains, lean protein',
                    'limit_alcohol': 'Max 1 drink/day (women) or 2 drinks/day (men)'
                },
                'activity': {
                    'goal': '150 minutes moderate aerobic activity per week',
                    'benefit': 'Can lower BP by 5-8 mmHg'
                },
                'weight': {
                    'goal': 'BMI 18.5-24.9 or 5-10% weight loss',
                    'benefit': 'Each kg lost can reduce BP by ~1 mmHg'
                }
            }
        }
        
        print(f"  ✓ Hypertension: {profile['sample_size']} participants")
        return profile
    
    def create_cvd_profile(self):
        """Profile of people with cardiovascular disease"""
        print("\nGenerating Cardiovascular Disease profile...")
        
        cvd = self.data[self.data['has_cvd'] == 1].copy()
        no_cvd = self.data[self.data['has_cvd'] == 0].copy()
        
        profile = {
            'disease': 'Cardiovascular Disease',
            'description': 'Population with heart attack, stroke, heart failure, or coronary disease',
            'sample_size': int(cvd['has_cvd'].sum()),
            
            'demographics': {
                'avg_age': float(cvd['age'].mean()),
                'age_std': float(cvd['age'].std()),
                'gender_distribution': cvd['gender'].value_counts().to_dict()
            },
            
            'clinical_markers': {
                'cholesterol': {
                    'total_mean': float(cvd['total_cholesterol'].mean()),
                    'target': '<200 mg/dL (total cholesterol)'
                },
                'blood_pressure': {
                    'systolic_mean': float(cvd['systolic_bp'].mean()),
                    'diastolic_mean': float(cvd['diastolic_bp'].mean()),
                    'target': '<120/80 mmHg'
                },
                'bmi': {
                    'mean': float(cvd['bmi'].mean()),
                    'target': '18.5-24.9 kg/m²'
                },
                'metabolic_risk': {
                    'avg_risk_score': float(cvd['metabolic_risk_score'].mean()),
                    'description': 'Score 0-5 based on glucose, BP, BMI, waist circumference'
                }
            },
            
            'lifestyle_patterns': {
                'dietary': {
                    'avg_sodium_mg': float(cvd['sodium_mg'].mean()),
                    'avg_saturated_fat_g': float(cvd['saturated_fat_g'].mean()),
                    'avg_fiber_g': float(cvd['fiber_g'].mean())
                },
                'activity': {
                    'avg_weekly_mins': float(cvd['weekly_activity_mins'].mean()),
                    'pct_meeting_guidelines': float(cvd['meets_activity_guidelines'].mean() * 100)
                }
            },
            
            'comorbidities': {
                'pct_with_diabetes': float(cvd['has_diabetes'].mean() * 100),
                'pct_with_hypertension': float(cvd['has_hypertension'].mean() * 100),
                'avg_metabolic_risk': float(cvd['metabolic_risk_score'].mean())
            },
            
            'recommendations': {
                'cholesterol': {
                    'target': 'Total <200 mg/dL, LDL <100 mg/dL (or <70 if high risk)',
                    'action': 'Limit saturated fats, increase omega-3 fatty acids'
                },
                'blood_pressure': {
                    'target': '<120/80 mmHg',
                    'critical': 'Essential for secondary prevention'
                },
                'diet': {
                    'sodium_limit': '<1,500mg per day (stricter for CVD)',
                    'saturated_fat': '<7% of total calories',
                    'fiber_goal': '>30g per day',
                    'omega3': 'Include fatty fish 2x per week'
                },
                'activity': {
                    'goal': 'Start with 30 min moderate activity 5x/week',
                    'progression': 'Gradually increase as tolerated',
                    'caution': 'Consult physician before starting new program'
                },
                'lifestyle': {
                    'smoking': 'Quit immediately if smoking',
                    'alcohol': 'Limit or avoid alcohol',
                    'stress': 'Practice stress management techniques'
                }
            }
        }
        
        print(f"  ✓ CVD: {profile['sample_size']} participants")
        return profile
    
    def create_healthy_reference_profile(self):
        """Profile of healthy population (no major chronic diseases)"""
        print("\nGenerating Healthy Reference profile...")
        
        # Define healthy as: no diabetes, no hypertension, no CVD, BMI 18.5-30
        healthy = self.data[
            (self.data['has_diabetes'] == 0) &
            (self.data['has_hypertension'] == 0) &
            (self.data['has_cvd'] == 0) &
            (self.data['bmi'] >= 18.5) &
            (self.data['bmi'] < 30)
        ].copy()
        
        profile = {
            'disease': 'Healthy Reference',
            'description': 'Population without diabetes, hypertension, CVD, and normal BMI',
            'sample_size': len(healthy),
            
            'clinical_markers': {
                'glucose': {
                    'mean': float(healthy['glucose'].mean()),
                    'std': float(healthy['glucose'].std()),
                    'median': float(healthy['glucose'].median())
                },
                'blood_pressure': {
                    'systolic_mean': float(healthy['systolic_bp'].mean()),
                    'diastolic_mean': float(healthy['diastolic_bp'].mean())
                },
                'bmi': {
                    'mean': float(healthy['bmi'].mean()),
                    'std': float(healthy['bmi'].std())
                },
                'cholesterol': {
                    'mean': float(healthy['total_cholesterol'].mean())
                }
            },
            
            'lifestyle_patterns': {
                'dietary': {
                    'avg_calories': float(healthy['calories'].mean()),
                    'avg_sugar_g': float(healthy['sugar_g'].mean()),
                    'avg_fiber_g': float(healthy['fiber_g'].mean()),
                    'avg_sodium_mg': float(healthy['sodium_mg'].mean())
                },
                'activity': {
                    'avg_weekly_mins': float(healthy['weekly_activity_mins'].mean()),
                    'pct_meeting_guidelines': float(healthy['meets_activity_guidelines'].mean() * 100)
                }
            },
            
            'target_ranges': {
                'glucose': '70-99 mg/dL (fasting)',
                'blood_pressure': '<120/80 mmHg',
                'bmi': '18.5-24.9 kg/m²',
                'total_cholesterol': '<200 mg/dL',
                'activity': '≥150 min/week moderate intensity'
            }
        }
        
        print(f"  ✓ Healthy Reference: {profile['sample_size']} participants")
        return profile
    
    def generate_all_profiles(self):
        """Generate all disease profiles and save to JSON"""
        print("="*60)
        print("GENERATING DISEASE PROFILES")
        print("="*60)
        
        profiles = {
            'diabetes': self.create_diabetes_profile(),
            'hypertension': self.create_hypertension_profile(),
            'cardiovascular_disease': self.create_cvd_profile(),
            'healthy_reference': self.create_healthy_reference_profile()
        }
        
        # Create output directory
        output_dir = Path('data/profiles')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        output_path = output_dir / 'disease_profiles.json'
        with open(output_path, 'w') as f:
            json.dump(profiles, f, indent=2, default=str)
        
        print(f"\n✓ Saved all profiles to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("PROFILE SUMMARY")
        print("="*60)
        for disease_key, profile in profiles.items():
            print(f"\n{profile['disease']}:")
            print(f"  Sample size: {profile['sample_size']:,}")
            if 'demographics' in profile:
                print(f"  Avg age: {profile['demographics']['avg_age']:.1f} years")
        
        return profiles


# Usage
if __name__ == "__main__":
    print("Loading processed NHANES data...")
    data = pd.read_csv('data/processed/nhanes_2021_2023_integrated.csv')
    
    print(f"Dataset: {len(data):,} participants\n")
    
    # Generate profiles
    generator = DiseaseProfileGenerator(data)
    profiles = generator.generate_all_profiles()
    
    print("\n" + "="*60)
    print("✅ PROFILE GENERATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review disease_profiles.json")
    print("2. Build recommendation_engine.py")
    print("3. Integrate with MIMIC chatbot")