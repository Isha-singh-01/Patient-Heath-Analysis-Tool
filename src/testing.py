"""
Comprehensive Testing Suite for Patient Health Analysis System

Tests:
1. System Performance (response time, accuracy)
2. Component Integration (model, NHANES, LLM)
3. Edge Cases (missing data, extreme values)
4. User Personas (25 diverse test cases)
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import system
import sys
import os
sys.path.append('src')
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


class SystemTester:
    """Comprehensive testing framework"""
    
    def __init__(self, model_dir='models_fixed', use_llm=True):
        self.results = {
            'performance': [],
            'edge_cases': [],
            'personas': [],
            'component_tests': [],
            'errors': []
        }
        
        print("="*70)
        print("INITIALIZING SYSTEM TESTER")
        print("="*70)
        
        try:
            # Load env if available
            try:
                from load_env import load_env, get_config
                load_env()
                config = get_config()
                
                # Use token from .env if available
                if config['hf_token'] and use_llm:
                    print(f"✅ Found HF_TOKEN in environment")
                    hf_token = config['hf_token']
                    use_llm = True
                else:
                    hf_token = None
                    use_llm = False
            except:
                hf_token = None
                use_llm = False
            
            self.system = ComprehensiveRecommendationSystem(
                model_dir=model_dir,
                profiles_path=f'{model_dir}/comprehensive_disease_profiles.json',
                use_llm=use_llm,
                hf_api_key=hf_token
            )
            print("✅ System loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load system: {e}")
            raise
    
    def test_performance(self, n_iterations=10):
        """Test response time performance"""
        print("\n" + "="*70)
        print("TEST 1: PERFORMANCE TESTING")
        print("="*70)
        
        test_input = {
            'age': 55,
            'gender': 'M',
            'symptoms_text': 'chest pain shortness of breath fatigue',
            'glucose': 120,
            'systolic_bp': 140,
            'bmi': 28.5
        }
        
        response_times = []
        
        for i in range(n_iterations):
            start_time = time.time()
            
            try:
                report = self.system.generate_comprehensive_report(test_input)
                elapsed = time.time() - start_time
                response_times.append(elapsed)
                
                print(f"  Run {i+1}/{n_iterations}: {elapsed:.3f}s")
                
            except Exception as e:
                self.results['errors'].append({
                    'test': 'performance',
                    'iteration': i+1,
                    'error': str(e)
                })
        
        # Calculate statistics
        if response_times:
            avg_time = np.mean(response_times)
            min_time = np.min(response_times)
            max_time = np.max(response_times)
            p95_time = np.percentile(response_times, 95)
            
            performance_result = {
                'test': 'response_time',
                'iterations': n_iterations,
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'p95_time': p95_time,
                'target': 2.0,
                'passed': avg_time < 2.0
            }
            
            self.results['performance'].append(performance_result)
            
            print(f"\n{'='*70}")
            print(f"Performance Summary:")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")
            print(f"  95th percentile: {p95_time:.3f}s")
            print(f"  Target: <2.0s")
            print(f"  Status: {'✅ PASS' if performance_result['passed'] else '❌ FAIL'}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n" + "="*70)
        print("TEST 2: EDGE CASE TESTING")
        print("="*70)
        
        edge_cases = [
            {
                'name': 'Missing all data',
                'input': {'age': 45, 'gender': 'M', 'symptoms_text': ''}
            },
            {
                'name': 'Only symptoms, no labs',
                'input': {
                    'age': 60,
                    'gender': 'F',
                    'symptoms_text': 'headache dizziness'
                }
            },
            {
                'name': 'Extreme glucose value',
                'input': {
                    'age': 55,
                    'gender': 'M',
                    'symptoms_text': 'confusion shakiness',
                    'glucose': 500
                }
            },
            {
                'name': 'Extreme low glucose',
                'input': {
                    'age': 55,
                    'gender': 'M',
                    'symptoms_text': 'weakness shakiness',
                    'glucose': 30
                }
            },
            {
                'name': 'Extremely high BP',
                'input': {
                    'age': 70,
                    'gender': 'M',
                    'symptoms_text': 'severe headache',
                    'systolic_bp': 220,
                    'diastolic_bp': 130
                }
            },
            {
                'name': 'Very high creatinine (kidney failure)',
                'input': {
                    'age': 65,
                    'gender': 'F',
                    'symptoms_text': 'fatigue nausea',
                    'creatinine': 8.5
                }
            },
            {
                'name': 'Unclear symptoms',
                'input': {
                    'age': 40,
                    'gender': 'M',
                    'symptoms_text': 'just not feeling well'
                }
            },
            {
                'name': 'Multiple symptoms mixed',
                'input': {
                    'age': 58,
                    'gender': 'F',
                    'symptoms_text': 'chest pain fatigue dizziness shortness of breath nausea headache',
                    'glucose': 180,
                    'systolic_bp': 160,
                    'creatinine': 2.1
                }
            },
            {
                'name': 'Age extremes - very young adult',
                'input': {
                    'age': 18,
                    'gender': 'M',
                    'symptoms_text': 'chest pain rapid heartbeat',
                    'systolic_bp': 140
                }
            },
            {
                'name': 'Age extremes - very elderly',
                'input': {
                    'age': 95,
                    'gender': 'F',
                    'symptoms_text': 'fatigue confusion weakness',
                    'glucose': 95,
                    'creatinine': 1.8
                }
            }
        ]
        
        for i, case in enumerate(edge_cases, 1):
            print(f"\n[{i}/{len(edge_cases)}] Testing: {case['name']}")
            
            start_time = time.time()
            passed = False
            error_msg = None
            
            try:
                report = self.system.generate_comprehensive_report(case['input'])
                elapsed = time.time() - start_time
                
                # Validate report structure
                assert 'prediction' in report, "Missing prediction"
                assert 'recommendations' in report, "Missing recommendations"
                assert report['prediction']['disease'], "No disease predicted"
                
                passed = True
                print(f"  ✅ Passed in {elapsed:.3f}s")
                print(f"     Predicted: {report['prediction']['disease']}")
                
            except Exception as e:
                elapsed = time.time() - start_time
                error_msg = str(e)
                print(f"  ❌ Failed: {error_msg}")
                
                self.results['errors'].append({
                    'test': 'edge_case',
                    'case': case['name'],
                    'error': error_msg
                })
            
            self.results['edge_cases'].append({
                'case': case['name'],
                'input': case['input'],
                'passed': passed,
                'response_time': elapsed,
                'error': error_msg
            })
    
    def test_component_integration(self):
        """Test individual components"""
        print("\n" + "="*70)
        print("TEST 3: COMPONENT INTEGRATION")
        print("="*70)
        
        test_input = {
            'age': 55,
            'gender': 'M',
            'symptoms_text': 'frequent urination increased thirst',
            'glucose': 145,
            'bmi': 31
        }
        
        components_tested = []
        
        # Test 1: Disease Prediction
        print("\n[1] Testing Disease Prediction...")
        try:
            features = self.system.prepare_input_features(test_input)
            probabilities = self.system.rf_model.predict_proba(features)[0]
            
            assert probabilities.sum() > 0.99, "Probabilities don't sum to 1"
            assert len(probabilities) == len(self.system.disease_names), "Probability count mismatch"
            
            print("  ✅ Disease prediction working")
            components_tested.append(('disease_prediction', True, None))
        except Exception as e:
            print(f"  ❌ Disease prediction failed: {e}")
            components_tested.append(('disease_prediction', False, str(e)))
        
        # Test 2: NHANES Profile Retrieval
        print("\n[2] Testing NHANES Profile Retrieval...")
        try:
            profile = self.system.get_nhanes_profile('Diabetes')
            
            assert 'sample_size' in profile, "Missing sample_size"
            assert profile['sample_size'] > 0, "Invalid sample size"
            
            print("  ✅ NHANES profile retrieval working")
            components_tested.append(('nhanes_profile', True, None))
        except Exception as e:
            print(f"  ❌ NHANES profile failed: {e}")
            components_tested.append(('nhanes_profile', False, str(e)))
        
        # Test 3: Population Comparison
        print("\n[3] Testing Population Comparison...")
        try:
            comparisons = self.system.compare_user_to_population(test_input, profile)
            
            assert 'glucose' in comparisons, "Glucose comparison missing"
            assert 'status' in comparisons['glucose'], "Status missing"
            
            print("  ✅ Population comparison working")
            components_tested.append(('population_comparison', True, None))
        except Exception as e:
            print(f"  ❌ Population comparison failed: {e}")
            components_tested.append(('population_comparison', False, str(e)))
        
        # Test 4: Lifestyle Context
        print("\n[4] Testing Lifestyle Context Engine...")
        try:
            from nhanes_lifestyle_engine import UserMetrics
            
            metrics = UserMetrics(
                age=55,
                glucose=145,
                bmi=31,
                sugar_g_day=80,
                activity_minutes_week=30
            )
            
            context = self.system.lifestyle_recommender.build_lifestyle_context('Diabetes', metrics)
            
            assert 'metrics' in context, "Missing metrics"
            assert 'notes' in context, "Missing notes"
            
            print("  ✅ Lifestyle context working")
            components_tested.append(('lifestyle_context', True, None))
        except Exception as e:
            print(f"  ❌ Lifestyle context failed: {e}")
            components_tested.append(('lifestyle_context', False, str(e)))
        
        # Test 5: Recommendation Generation
        print("\n[5] Testing Recommendation Generation...")
        try:
            recommendations = self.system._generate_fallback_recommendations(
                'Diabetes', profile, comparisons, 
                {'Diabetes': 0.85, 'Hypertension': 0.10, 'Heart Failure': 0.05},
                context, test_input
            )
            
            assert len(recommendations) > 100, "Recommendations too short"
            assert "### 1. Disease risk estimation" in recommendations, "Missing section 1"
            assert "### 2. Lifestyle review" in recommendations, "Missing section 2"
            
            print("  ✅ Recommendation generation working")
            components_tested.append(('recommendations', True, None))
        except Exception as e:
            print(f"  ❌ Recommendation generation failed: {e}")
            components_tested.append(('recommendations', False, str(e)))
        
        self.results['component_tests'] = components_tested
        
        # Summary
        passed = sum(1 for _, status, _ in components_tested if status)
        total = len(components_tested)
        print(f"\n{'='*70}")
        print(f"Component Integration: {passed}/{total} passed")
    
    def test_diverse_personas(self):
        """Test 25 diverse patient personas"""
        print("\n" + "="*70)
        print("TEST 4: DIVERSE PATIENT PERSONAS (25 cases)")
        print("="*70)
        
        personas = self._generate_test_personas()
        
        for i, persona in enumerate(personas, 1):
            print(f"\n[{i}/25] Testing: {persona['name']}")
            print(f"  Profile: {persona['age']}y {persona['gender']}, {persona['risk_category']}")
            
            start_time = time.time()
            passed = False
            error_msg = None
            predicted_disease = None
            
            try:
                report = self.system.generate_comprehensive_report(persona['input'])
                elapsed = time.time() - start_time
                
                predicted_disease = report['prediction']['disease']
                confidence = report['prediction']['confidence']
                
                # Validate output
                assert report['recommendations'], "No recommendations generated"
                assert len(report['prediction']['top_3_predictions']) == 3, "Top 3 missing"
                
                passed = True
                print(f"  ✅ Passed in {elapsed:.3f}s")
                print(f"     Predicted: {predicted_disease} ({confidence*100:.1f}%)")
                
            except Exception as e:
                elapsed = time.time() - start_time
                error_msg = str(e)
                print(f"  ❌ Failed: {error_msg}")
            
            # Store results
            self.results['personas'].append({
                'persona_name': persona['name'],
                'age': persona['age'],
                'gender': persona['gender'],
                'risk_category': persona['risk_category'],
                'expected_diseases': persona.get('expected_diseases', []),
                'predicted_disease': predicted_disease,
                'passed': passed,
                'response_time': elapsed,
                'error': error_msg
            })
    
    def _generate_test_personas(self) -> List[Dict]:
        """Generate 25 diverse test personas"""
        
        personas = [
            # ===== DIABETES PERSONAS (5) =====
            {
                'name': 'Diabetes - Classic presentation',
                'age': 55, 'gender': 'M', 'risk_category': 'High Risk',
                'expected_diseases': ['Diabetes', 'Diabetes Mellitus'],
                'input': {
                    'age': 55, 'gender': 'M', 'bmi': 32.5,
                    'symptoms_text': 'increased thirst frequent urination fatigue blurred vision',
                    'glucose': 185, 'systolic_bp': 142,
                    'sugar_g_day': 120, 'activity_minutes_week': 20, 'fiber_g_day': 10
                }
            },
            {
                'name': 'Diabetes - Prediabetic',
                'age': 48, 'gender': 'F', 'risk_category': 'Medium Risk',
                'input': {
                    'age': 48, 'gender': 'F', 'bmi': 27.5,
                    'symptoms_text': 'fatigue increased hunger',
                    'glucose': 115, 'systolic_bp': 128,
                    'sugar_g_day': 85, 'activity_minutes_week': 60
                }
            },
            {
                'name': 'Diabetes - Type 2 with complications',
                'age': 62, 'gender': 'M', 'risk_category': 'High Risk',
                'input': {
                    'age': 62, 'gender': 'M', 'bmi': 35.0,
                    'symptoms_text': 'numbness in feet blurred vision frequent urination',
                    'glucose': 220, 'creatinine': 1.4, 'systolic_bp': 155,
                    'sugar_g_day': 140, 'activity_minutes_week': 0
                }
            },
            {
                'name': 'Diabetes - Well-controlled',
                'age': 58, 'gender': 'F', 'risk_category': 'Low Risk',
                'input': {
                    'age': 58, 'gender': 'F', 'bmi': 24.5,
                    'symptoms_text': 'mild fatigue',
                    'glucose': 105, 'systolic_bp': 122,
                    'sugar_g_day': 40, 'activity_minutes_week': 180, 'fiber_g_day': 30
                }
            },
            {
                'name': 'Diabetes - Young onset',
                'age': 32, 'gender': 'M', 'risk_category': 'Medium Risk',
                'input': {
                    'age': 32, 'gender': 'M', 'bmi': 29.0,
                    'symptoms_text': 'increased thirst weight loss',
                    'glucose': 165, 'sugar_g_day': 95, 'activity_minutes_week': 45
                }
            },
            
            # ===== HYPERTENSION PERSONAS (5) =====
            {
                'name': 'Hypertension - Stage 2',
                'age': 64, 'gender': 'M', 'risk_category': 'High Risk',
                'expected_diseases': ['Hypertension', 'Essential Hypertension'],
                'input': {
                    'age': 64, 'gender': 'M', 'bmi': 30.0,
                    'symptoms_text': 'headache dizziness chest discomfort',
                    'systolic_bp': 165, 'diastolic_bp': 102,
                    'sodium_mg_day': 4200, 'activity_minutes_week': 30
                }
            },
            {
                'name': 'Hypertension - Stage 1',
                'age': 52, 'gender': 'F', 'risk_category': 'Medium Risk',
                'input': {
                    'age': 52, 'gender': 'F', 'bmi': 27.8,
                    'symptoms_text': 'occasional headaches',
                    'systolic_bp': 138, 'diastolic_bp': 88,
                    'sodium_mg_day': 3200, 'activity_minutes_week': 90
                }
            },
            {
                'name': 'Hypertension - Elderly',
                'age': 78, 'gender': 'F', 'risk_category': 'High Risk',
                'input': {
                    'age': 78, 'gender': 'F', 'bmi': 26.0,
                    'symptoms_text': 'dizziness fatigue',
                    'systolic_bp': 158, 'diastolic_bp': 92,
                    'sodium_mg_day': 3500, 'activity_minutes_week': 50
                }
            },
            {
                'name': 'Hypertension - With diabetes',
                'age': 60, 'gender': 'M', 'risk_category': 'High Risk',
                'input': {
                    'age': 60, 'gender': 'M', 'bmi': 33.0,
                    'symptoms_text': 'headache blurred vision frequent urination',
                    'glucose': 156, 'systolic_bp': 148, 'diastolic_bp': 94,
                    'sodium_mg_day': 3800, 'sugar_g_day': 100
                }
            },
            {
                'name': 'Hypertension - Pre-hypertensive',
                'age': 45, 'gender': 'M', 'risk_category': 'Low Risk',
                'input': {
                    'age': 45, 'gender': 'M', 'bmi': 26.5,
                    'symptoms_text': 'mild headaches stress',
                    'systolic_bp': 128, 'diastolic_bp': 82,
                    'sodium_mg_day': 2800, 'activity_minutes_week': 120
                }
            },
            
            # ===== HEART FAILURE PERSONAS (5) =====
            {
                'name': 'Heart Failure - Severe',
                'age': 72, 'gender': 'M', 'risk_category': 'High Risk',
                'expected_diseases': ['Heart Failure', 'Congestive Heart Failure'],
                'input': {
                    'age': 72, 'gender': 'M', 'bmi': 28.0,
                    'symptoms_text': 'severe shortness of breath swelling in legs fatigue',
                    'systolic_bp': 145, 'creatinine': 1.6,
                    'sodium_mg_day': 4000, 'activity_minutes_week': 10
                }
            },
            {
                'name': 'Heart Failure - Compensated',
                'age': 68, 'gender': 'F', 'risk_category': 'Medium Risk',
                'input': {
                    'age': 68, 'gender': 'F', 'bmi': 25.5,
                    'symptoms_text': 'shortness of breath on exertion mild fatigue',
                    'systolic_bp': 132, 'creatinine': 1.1,
                    'sodium_mg_day': 2100, 'activity_minutes_week': 75
                }
            },
            {
                'name': 'Heart Failure - With AFib',
                'age': 75, 'gender': 'M', 'risk_category': 'High Risk',
                'input': {
                    'age': 75, 'gender': 'M', 'bmi': 27.0,
                    'symptoms_text': 'irregular heartbeat shortness of breath fatigue',
                    'systolic_bp': 138, 'potassium': 3.8,
                    'activity_minutes_week': 40
                }
            },
            {
                'name': 'Heart Failure - Post-MI',
                'age': 66, 'gender': 'M', 'risk_category': 'High Risk',
                'input': {
                    'age': 66, 'gender': 'M', 'bmi': 29.5,
                    'symptoms_text': 'chest pain shortness of breath swelling',
                    'systolic_bp': 142, 'glucose': 128,
                    'sodium_mg_day': 3400, 'activity_minutes_week': 25
                }
            },
            {
                'name': 'Heart Failure - Young presentation',
                'age': 52, 'gender': 'M', 'risk_category': 'Medium Risk',
                'input': {
                    'age': 52, 'gender': 'M', 'bmi': 31.0,
                    'symptoms_text': 'shortness of breath fatigue chest discomfort',
                    'systolic_bp': 135, 'activity_minutes_week': 45
                }
            },
            
            # ===== KIDNEY FAILURE PERSONAS (5) =====
            {
                'name': 'Kidney Failure - Advanced CKD',
                'age': 70, 'gender': 'M', 'risk_category': 'High Risk',
                'expected_diseases': ['Kidney Failure', 'Chronic Kidney Disease'],
                'input': {
                    'age': 70, 'gender': 'M', 'bmi': 28.0,
                    'symptoms_text': 'fatigue nausea decreased urine output swelling',
                    'creatinine': 3.8, 'urea_nitrogen': 65, 'potassium': 5.5,
                    'protein_g_day': 120, 'sodium_mg_day': 4500
                }
            },
            {
                'name': 'Kidney Failure - Stage 3 CKD',
                'age': 65, 'gender': 'F', 'risk_category': 'Medium Risk',
                'input': {
                    'age': 65, 'gender': 'F', 'bmi': 26.5,
                    'symptoms_text': 'fatigue mild swelling',
                    'creatinine': 1.8, 'urea_nitrogen': 28,
                    'protein_g_day': 85, 'sodium_mg_day': 2800
                }
            },
            {
                'name': 'Kidney Failure - Diabetic nephropathy',
                'age': 62, 'gender': 'M', 'risk_category': 'High Risk',
                'input': {
                    'age': 62, 'gender': 'M', 'bmi': 32.0,
                    'symptoms_text': 'fatigue increased urination swelling feet',
                    'glucose': 195, 'creatinine': 2.4, 'systolic_bp': 152,
                    'sugar_g_day': 110, 'protein_g_day': 105
                }
            },
            {
                'name': 'Kidney Failure - Hypertensive nephropathy',
                'age': 68, 'gender': 'F', 'risk_category': 'High Risk',
                'input': {
                    'age': 68, 'gender': 'F', 'bmi': 29.0,
                    'symptoms_text': 'headache fatigue decreased urine',
                    'systolic_bp': 168, 'creatinine': 2.1, 'urea_nitrogen': 42,
                    'sodium_mg_day': 4100
                }
            },
            {
                'name': 'Kidney Failure - Early CKD',
                'age': 58, 'gender': 'M', 'risk_category': 'Low Risk',
                'input': {
                    'age': 58, 'gender': 'M', 'bmi': 27.5,
                    'symptoms_text': 'mild fatigue',
                    'creatinine': 1.3, 'glucose': 108,
                    'protein_g_day': 95, 'activity_minutes_week': 100
                }
            },
            
            # ===== MIXED/OTHER CONDITIONS (5) =====
            {
                'name': 'Asthma - Moderate',
                'age': 42, 'gender': 'F', 'risk_category': 'Medium Risk',
                'input': {
                    'age': 42, 'gender': 'F', 'bmi': 24.0,
                    'symptoms_text': 'wheezing shortness of breath chest tightness',
                    'activity_minutes_week': 110
                }
            },
            {
                'name': 'Healthy - Active lifestyle',
                'age': 35, 'gender': 'M', 'risk_category': 'Low Risk',
                'input': {
                    'age': 35, 'gender': 'M', 'bmi': 23.5,
                    'symptoms_text': 'occasional fatigue',
                    'glucose': 92, 'systolic_bp': 115,
                    'sugar_g_day': 35, 'activity_minutes_week': 240, 'fiber_g_day': 32
                }
            },
            {
                'name': 'Multiple conditions - Elderly',
                'age': 82, 'gender': 'F', 'risk_category': 'High Risk',
                'input': {
                    'age': 82, 'gender': 'F', 'bmi': 24.0,
                    'symptoms_text': 'fatigue dizziness confusion weakness',
                    'glucose': 142, 'systolic_bp': 158, 'creatinine': 1.7,
                    'activity_minutes_week': 30
                }
            },
            {
                'name': 'Obesity - Sedentary lifestyle',
                'age': 44, 'gender': 'M', 'risk_category': 'High Risk',
                'input': {
                    'age': 44, 'gender': 'M', 'bmi': 38.5,
                    'symptoms_text': 'joint pain fatigue shortness of breath',
                    'glucose': 118, 'systolic_bp': 136,
                    'sugar_g_day': 135, 'activity_minutes_week': 5, 'fiber_g_day': 8
                }
            },
            {
                'name': 'UTI - Uncomplicated',
                'age': 38, 'gender': 'F', 'risk_category': 'Low Risk',
                'input': {
                    'age': 38, 'gender': 'F', 'bmi': 22.5,
                    'symptoms_text': 'burning urination frequent urination',
                    'glucose': 95, 'activity_minutes_week': 150
                }
            }
        ]
        
        return personas
    
    def generate_report(self, output_dir='test_results'):
        """Generate comprehensive test report"""
        print("\n" + "="*70)
        print("GENERATING TEST REPORT")
        print("="*70)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Performance Summary
        performance_df = pd.DataFrame(self.results['performance'])
        if not performance_df.empty:
            perf_file = output_path / f'performance_results_{timestamp}.csv'
            performance_df.to_csv(perf_file, index=False)
            print(f"✅ Performance results: {perf_file}")
        
        # Edge Cases
        edge_df = pd.DataFrame(self.results['edge_cases'])
        if not edge_df.empty:
            edge_file = output_path / f'edge_case_results_{timestamp}.csv'
            edge_df.to_csv(edge_file, index=False)
            print(f"✅ Edge case results: {edge_file}")
        
        # Personas
        persona_df = pd.DataFrame(self.results['personas'])
        if not persona_df.empty:
            persona_file = output_path / f'persona_results_{timestamp}.csv'
            persona_df.to_csv(persona_file, index=False)
            print(f"✅ Persona results: {persona_file}")
        
        # Component Tests
        component_df = pd.DataFrame(self.results['component_tests'], 
                                    columns=['component', 'passed', 'error'])
        comp_file = output_path / f'component_results_{timestamp}.csv'
        component_df.to_csv(comp_file, index=False)
        print(f"✅ Component results: {comp_file}")
        
        # Errors
        if self.results['errors']:
            error_df = pd.DataFrame(self.results['errors'])
            error_file = output_path / f'errors_{timestamp}.csv'
            error_df.to_csv(error_file, index=False)
            print(f"⚠️  Errors logged: {error_file}")
        
        # Summary report
        self._generate_summary_report(output_path, timestamp)
    
    def _generate_summary_report(self, output_path, timestamp):
        """Generate human-readable summary"""
        
        summary_file = output_path / f'TEST_SUMMARY_{timestamp}.txt'
        
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PATIENT HEALTH ANALYSIS SYSTEM - TEST REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Performance
            f.write("1. PERFORMANCE TESTING\n")
            f.write("-"*70 + "\n")
            if self.results['performance']:
                perf = self.results['performance'][0]
                f.write(f"Average Response Time: {perf['avg_time']:.3f}s\n")
                f.write(f"95th Percentile: {perf['p95_time']:.3f}s\n")
                f.write(f"Target: <{perf['target']}s\n")
                f.write(f"Status: {'PASS ✅' if perf['passed'] else 'FAIL ❌'}\n\n")
            
            # Edge Cases
            f.write("2. EDGE CASE TESTING\n")
            f.write("-"*70 + "\n")
            edge_passed = sum(1 for e in self.results['edge_cases'] if e['passed'])
            edge_total = len(self.results['edge_cases'])
            f.write(f"Passed: {edge_passed}/{edge_total} ({edge_passed/edge_total*100:.1f}%)\n\n")
            
            # Components
            f.write("3. COMPONENT INTEGRATION\n")
            f.write("-"*70 + "\n")
            comp_passed = sum(1 for _, status, _ in self.results['component_tests'] if status)
            comp_total = len(self.results['component_tests'])
            f.write(f"Passed: {comp_passed}/{comp_total} ({comp_passed/comp_total*100:.1f}%)\n\n")
            
            # Personas
            f.write("4. DIVERSE PERSONAS\n")
            f.write("-"*70 + "\n")
            persona_passed = sum(1 for p in self.results['personas'] if p['passed'])
            persona_total = len(self.results['personas'])
            f.write(f"Passed: {persona_passed}/{persona_total} ({persona_passed/persona_total*100:.1f}%)\n\n")
            
            # Errors
            if self.results['errors']:
                f.write("5. ERRORS\n")
                f.write("-"*70 + "\n")
                for error in self.results['errors']:
                    f.write(f"  • {error['test']}: {error.get('case', 'N/A')} - {error['error']}\n")
                f.write("\n")
            
            # Overall
            f.write("="*70 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("="*70 + "\n")
            total_tests = edge_total + comp_total + persona_total
            total_passed = edge_passed + comp_passed + persona_passed
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)\n")
            f.write(f"Failed: {total_tests - total_passed}\n")
            f.write(f"Errors: {len(self.results['errors'])}\n")
        
        print(f"✅ Summary report: {summary_file}")
        
        # Print summary to console
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
        print(f"Failed: {total_tests - total_passed}")
        print(f"Errors: {len(self.results['errors'])}")
        print("="*70)


def run_full_test_suite():
    """Run complete test suite"""
    
    print("\n" + "="*70)
    print("PATIENT HEALTH ANALYSIS SYSTEM - FULL TEST SUITE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    tester = SystemTester(model_dir='models_fixed')
    
    # Run all tests
    tester.test_performance(n_iterations=5)
    tester.test_edge_cases()
    tester.test_component_integration()
    tester.test_diverse_personas()
    
    # Generate reports
    tester.generate_report()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)
    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults saved to: test_results/")


if __name__ == "__main__":
    run_full_test_suite()