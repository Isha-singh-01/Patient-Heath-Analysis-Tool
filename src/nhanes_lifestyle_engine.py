"""
Enhanced NHANES Lifestyle Engine
Combines age-banded population statistics with clinical guidelines
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class UserMetrics:
    """User's lifestyle and health metrics"""
    age: int
    gender: Optional[str] = None
    bmi: Optional[float] = None
    glucose: Optional[float] = None
    sodium_mg_day: Optional[float] = None
    sugar_g_day: Optional[float] = None
    activity_minutes_week: Optional[float] = None
    smoking_status: Optional[str] = None
    fiber_g_day: Optional[float] = None
    protein_g_day: Optional[float] = None
    alcohol_g_day: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None


class EnhancedLifestyleRecommender:
    """
    Generates structured lifestyle context with:
    - Age-banded NHANES population statistics
    - Clinical guideline thresholds
    - Intermediate targets for gradual improvement
    - Status classification (high/low/ok/unknown)
    """
    
    def __init__(self, guidelines_path: Optional[Path] = None):
        # Define age bands
        self.age_bands = {
            "18-29": (18, 29),
            "30-44": (30, 44),
            "45-59": (45, 59),
            "60+": (60, 150)
        }
        
        # Clinical guidelines (AHA, ADA, WHO, CDC)
        self.guidelines = {
            "sugar": {
                "who_limit_g": 50,  # WHO: <10% of daily calories (~50g for 2000 cal diet)
                "unit": "g/day"
            },
            "sodium": {
                "limit_mg": 2300,  # AHA: <2300mg/day, ideal <1500mg
                "ideal_mg": 1500,
                "unit": "mg/day"
            },
            "activity": {
                "min_minutes_week": 150,  # CDC: ≥150 min/week moderate intensity
                "unit": "min/week"
            },
            "bmi": {
                "healthy_range": (18.5, 24.9),
                "overweight": (25.0, 29.9),
                "obese": 30.0
            },
            "fiber": {
                "min_g": 25,  # Dietary Guidelines: 25-38g/day
                "max_g": 38,
                "unit": "g/day"
            },
            "protein": {
                "min_g": 50,  # ~0.8 g/kg for 60kg person
                "max_g": 120,  # ~2.0 g/kg for active individuals
                "unit": "g/day"
            },
            "alcohol": {
                "max_g": 14,  # ~1 drink/day for women, 2 for men (averaged)
                "unit": "g/day"
            },
            "glucose": {
                "normal_max": 100,  # Fasting glucose <100 mg/dL
                "prediabetic": 126,
                "unit": "mg/dL"
            },
            "blood_pressure": {
                "normal_systolic": 120,
                "elevated_systolic": 130,
                "hypertensive_systolic": 140,
                "unit": "mmHg"
            }
        }
        
        # NHANES age-banded population means (example data)
        self.nhanes_population_means = {
            "18-29": {
                "sodium_mg": 3400,
                "sugar_g": 77,
                "fiber_g": 16,
                "protein_g": 85,
                "alcohol_g": 8,
                "activity_min": 180
            },
            "30-44": {
                "sodium_mg": 3600,
                "sugar_g": 70,
                "fiber_g": 17,
                "protein_g": 90,
                "alcohol_g": 10,
                "activity_min": 140
            },
            "45-59": {
                "sodium_mg": 3500,
                "sugar_g": 65,
                "fiber_g": 18,
                "protein_g": 85,
                "alcohol_g": 12,
                "activity_min": 120
            },
            "60+": {
                "sodium_mg": 3200,
                "sugar_g": 55,
                "fiber_g": 19,
                "protein_g": 75,
                "alcohol_g": 8,
                "activity_min": 100
            }
        }
    
    def get_age_band(self, age: int) -> str:
        """Map age to NHANES band"""
        for band, (low, high) in self.age_bands.items():
            if low <= age <= high:
                return band
        return "60+"  # Default for outliers
    
    def calculate_status(self, user_value: Optional[float], 
                        guideline: Optional[float],
                        comparison_type: str = "high_if_above") -> str:
        """
        Determine status: high, low, ok, unknown
        comparison_type: 'high_if_above', 'low_if_below'
        """
        if user_value is None or guideline is None:
            return "unknown"
        
        if comparison_type == "high_if_above":
            if user_value > guideline * 1.1:
                return "high"
            elif user_value < guideline * 0.9:
                return "low"
            return "ok"
        else:  # low_if_below
            if user_value < guideline * 0.9:
                return "low"
            elif user_value > guideline * 1.1:
                return "high"
            return "ok"
    
    def calculate_intermediate_target(self, user: float, 
                                     guideline: float,
                                     status: str) -> Optional[float]:
        """Calculate realistic intermediate goal (halfway to guideline)"""
        if status in ["high", "low"]:
            return round((user + guideline) / 2, 1)
        return None
    
    def build_lifestyle_context(self, disease: str, 
                               metrics: UserMetrics) -> Dict[str, Any]:
        """
        Build comprehensive lifestyle context for LLM
        
        Returns structured dict with:
        - disease (reference only, not diagnosis)
        - age, age_band
        - metrics: detailed breakdown of each health metric
        - notes: list of flags (sugar_high, activity_low, etc.)
        """
        age_band = self.get_age_band(metrics.age)
        pop_stats = self.nhanes_population_means.get(age_band, {})
        
        context = {
            "disease": disease,
            "age": metrics.age,
            "age_band": age_band,
            "metrics": {},
            "notes": []
        }
        
        # Sugar
        if metrics.sugar_g_day is not None:
            guideline = self.guidelines["sugar"]["who_limit_g"]
            status = self.calculate_status(metrics.sugar_g_day, guideline, "high_if_above")
            target = self.calculate_intermediate_target(metrics.sugar_g_day, guideline, status)
            
            if status == "high":
                context["notes"].append("sugar_high")
            
            context["metrics"]["sugar"] = {
                "user": metrics.sugar_g_day,
                "population_mean": pop_stats.get("sugar_g"),
                "guideline": guideline,
                "intermediate_target": target,
                "unit": "g/day",
                "status": status
            }
        
        # Sodium
        if metrics.sodium_mg_day is not None:
            guideline = self.guidelines["sodium"]["limit_mg"]
            status = self.calculate_status(metrics.sodium_mg_day, guideline, "high_if_above")
            target = self.calculate_intermediate_target(metrics.sodium_mg_day, guideline, status)
            
            if status == "high":
                context["notes"].append("sodium_high")
            
            context["metrics"]["sodium"] = {
                "user": metrics.sodium_mg_day,
                "population_mean": pop_stats.get("sodium_mg"),
                "guideline": guideline,
                "ideal": self.guidelines["sodium"]["ideal_mg"],
                "intermediate_target": target,
                "unit": "mg/day",
                "status": status
            }
        
        # Physical Activity
        if metrics.activity_minutes_week is not None:
            guideline = self.guidelines["activity"]["min_minutes_week"]
            status = self.calculate_status(metrics.activity_minutes_week, guideline, "low_if_below")
            target = self.calculate_intermediate_target(metrics.activity_minutes_week, guideline, status)
            
            if status == "low":
                context["notes"].append("activity_low")
            
            context["metrics"]["activity"] = {
                "user": metrics.activity_minutes_week,
                "population_mean": pop_stats.get("activity_min"),
                "guideline_min": guideline,
                "intermediate_target": target,
                "unit": "min/week",
                "status": status
            }
        
        # BMI
        if metrics.bmi is not None:
            low, high = self.guidelines["bmi"]["healthy_range"]
            if metrics.bmi < low:
                status = "underweight"
            elif metrics.bmi <= high:
                status = "healthy_range"
            elif metrics.bmi < self.guidelines["bmi"]["obese"]:
                status = "overweight"
            else:
                status = "obese"
            
            context["notes"].append(f"bmi_{status}")
            context["metrics"]["bmi"] = {
                "user": metrics.bmi,
                "healthy_range": [low, high],
                "status": status
            }
        
        # Fiber
        if metrics.fiber_g_day is not None:
            guideline = self.guidelines["fiber"]["min_g"]
            status = self.calculate_status(metrics.fiber_g_day, guideline, "low_if_below")
            target = self.calculate_intermediate_target(metrics.fiber_g_day, guideline, status)
            
            if status == "low":
                context["notes"].append("fiber_low")
            
            context["metrics"]["fiber"] = {
                "user": metrics.fiber_g_day,
                "population_mean": pop_stats.get("fiber_g"),
                "guideline_min": guideline,
                "intermediate_target": target,
                "unit": "g/day",
                "status": status
            }
        
        # Protein
        if metrics.protein_g_day is not None:
            min_g = self.guidelines["protein"]["min_g"]
            max_g = self.guidelines["protein"]["max_g"]
            
            if metrics.protein_g_day < min_g * 0.9:
                status = "low"
                context["notes"].append("protein_low")
            elif metrics.protein_g_day > max_g * 2.0:
                status = "very_high"
                context["notes"].append("protein_very_high")
            else:
                status = "ok"
            
            target = self.calculate_intermediate_target(metrics.protein_g_day, min_g, status) if status == "low" else None
            
            context["metrics"]["protein"] = {
                "user": metrics.protein_g_day,
                "population_mean": pop_stats.get("protein_g"),
                "guideline_min": min_g,
                "guideline_max_ref": max_g,
                "intermediate_target": target,
                "unit": "g/day",
                "status": status
            }
        
        # Alcohol
        if metrics.alcohol_g_day is not None:
            guideline = self.guidelines["alcohol"]["max_g"]
            status = self.calculate_status(metrics.alcohol_g_day, guideline, "high_if_above")
            target = self.calculate_intermediate_target(metrics.alcohol_g_day, guideline, status)
            
            if status == "high":
                context["notes"].append("alcohol_high")
            
            context["metrics"]["alcohol"] = {
                "user": metrics.alcohol_g_day,
                "population_mean": pop_stats.get("alcohol_g"),
                "guideline_max": guideline,
                "intermediate_target": target,
                "unit": "g/day",
                "status": status
            }
        
        # Smoking
        if metrics.smoking_status:
            status = metrics.smoking_status.lower()
            context["metrics"]["smoking_status"] = {"user": status}
            if status in ["current", "smoker"]:
                context["notes"].append("smoker")
        
        # Glucose
        if metrics.glucose is not None:
            normal = self.guidelines["glucose"]["normal_max"]
            prediabetic = self.guidelines["glucose"]["prediabetic"]
            
            if metrics.glucose >= prediabetic:
                status = "diabetic_range"
            elif metrics.glucose >= normal:
                status = "prediabetic_range"
            else:
                status = "normal"
            
            if status != "normal":
                context["notes"].append(f"glucose_{status}")
            
            context["metrics"]["glucose"] = {
                "user": metrics.glucose,
                "normal_max": normal,
                "prediabetic_threshold": prediabetic,
                "unit": "mg/dL",
                "status": status
            }
        
        # Blood Pressure
        if metrics.systolic_bp is not None:
            normal = self.guidelines["blood_pressure"]["normal_systolic"]
            elevated = self.guidelines["blood_pressure"]["elevated_systolic"]
            hypertensive = self.guidelines["blood_pressure"]["hypertensive_systolic"]
            
            if metrics.systolic_bp >= hypertensive:
                status = "stage_2_hypertension"
            elif metrics.systolic_bp >= elevated:
                status = "stage_1_hypertension"
            elif metrics.systolic_bp >= normal:
                status = "elevated"
            else:
                status = "normal"
            
            if status != "normal":
                context["notes"].append(f"bp_{status}")
            
            context["metrics"]["blood_pressure"] = {
                "user_systolic": metrics.systolic_bp,
                "user_diastolic": metrics.diastolic_bp,
                "normal_systolic": normal,
                "target": "<120/80 mmHg",
                "unit": "mmHg",
                "status": status
            }
        
        return context