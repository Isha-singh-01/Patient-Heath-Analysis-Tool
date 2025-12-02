# src/nhanes_engine.py

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

GUIDE_PATH = Path("data/nhanes_guidelines_2021_2023.json")

@dataclass
class UserProfile:
    age: int
    sex: Optional[str] = None
    sugar_g_day: Optional[float] = None
    sodium_mg_day: Optional[float] = None
    bmi: Optional[float] = None
    activity_minutes_week: Optional[float] = None
    smoking_status: Optional[str] = None
    fiber_g_day: Optional[float] = None
    protein_g_day: Optional[float] = None
    alcohol_g_day: Optional[float] = None

class LifestyleRecommender:
    def __init__(self, guidelines_path: Path = GUIDE_PATH):
        with open(guidelines_path) as f:
            data = json.load(f)
        self.nhanes_stats: List[Dict[str, Any]] = data["nhanes_stats"]
        self.guidelines: Dict[str, Any] = data["guidelines"]

    def _age_band(self, age: int) -> str:
        """
        Map numeric age to one of the bands:
        '0-17', '18-29', '30-44', '45-59', '60+'
        """
        if age <= 17:
            return "0-17"
        elif age <= 29:
            return "18-29"
        elif age <= 44:
            return "30-44"
        elif age <= 59:
            return "45-59"
        else:
            return "60+"

    def _lookup_stats(self, disease: str, age_band: str) -> Dict[str, Any]:
        for row in self.nhanes_stats:
            if row["disease_category"] == disease and row["age_band"] == age_band:
                return row
        # If we don't find a match, try a fallback ignoring age_band
        for row in self.nhanes_stats:
            if row["disease_category"] == disease:
                return row
        # Last fallback: empty dict
        return {}

    def build_context(self, disease: str, profile: UserProfile) -> Dict[str, Any]:
        """
        Returns a machine-friendly dict for the LLM.
        Contains NHANES numbers, guideline thresholds, and intermediate targets.
        """
        band = self._age_band(profile.age)
        stats = self._lookup_stats(disease, band)

        ctx: Dict[str, Any] = {
            "disease": disease,
            "age": profile.age,
            "age_band": band,
            "metrics": {},
            "notes": [],
        }

        # --- Sugar ---
        if profile.sugar_g_day is not None and stats:
            user = profile.sugar_g_day
            pop_mean = stats["sugar_mean"]
            guideline = self.guidelines["sugar"]["who_limit_g"]  # 50g realistic target
            target = None
            if user > guideline:
                target = max(guideline, (user + guideline) / 2)
                ctx["notes"].append("sugar_high")
            ctx["metrics"]["sugar"] = {
                "user": user,
                "population_mean": pop_mean,
                "guideline": guideline,
                "intermediate_target": target,
            }

        # --- Sodium ---
        if profile.sodium_mg_day is not None and stats:
            user = profile.sodium_mg_day
            pop_mean = stats["sodium_mean"]
            limit = self.guidelines["sodium"]["limit_mg"]
            target = None
            if user > limit:
                target = max(limit, (user + limit) / 2)
                ctx["notes"].append("sodium_high")
            ctx["metrics"]["sodium"] = {
                "user": user,
                "population_mean": pop_mean,
                "guideline": limit,
                "intermediate_target": target,
            }

        # --- Activity ---
        if profile.activity_minutes_week is not None and stats:
            user = profile.activity_minutes_week
            min_needed = self.guidelines["activity"]["min_minutes_week"]
            target = None
            if user < min_needed:
                target = (user + min_needed) / 2
                ctx["notes"].append("activity_low")
            ctx["metrics"]["activity"] = {
                "user": user,
                "guideline_min": min_needed,
                "intermediate_target": target,
            }

        # --- BMI ---
        if profile.bmi is not None and stats:
            user = profile.bmi
            low, high = self.guidelines["bmi"]["healthy_range"]
            if user < low:
                status = "underweight"
            elif user > high:
                status = "overweight_obese"
            else:
                status = "healthy_range"
            ctx["notes"].append(f"bmi_{status}")
            ctx["metrics"]["bmi"] = {
                "user": user,
                "healthy_range": [low, high],
                "status": status,
            }

        # --- Fiber ---
        if profile.fiber_g_day is not None and stats:
            user = profile.fiber_g_day
            pop_mean = stats["fiber_mean"]
            min_needed = self.guidelines["fiber"]["min_g"]  # 25g
            target = None
            if user < min_needed:
                target = (user + min_needed) / 2
                ctx["notes"].append("fiber_low")
            ctx["metrics"]["fiber"] = {
                "user": user,
                "population_mean": pop_mean,
                "guideline_min": min_needed,
                "intermediate_target": target,
            }

        # --- Protein ---
        if profile.protein_g_day is not None and stats:
            user = profile.protein_g_day
            pop_mean = stats["protein_mean"]
            min_needed = self.guidelines["protein"]["min_g"]  # 50g
            max_safe = self.guidelines["protein"]["max_g"]    # 60g
            target = None
            if user < min_needed:
                target = (user + min_needed) / 2
                ctx["notes"].append("protein_low")
            if user > max_safe * 1.8:  # very high intake
                ctx["notes"].append("protein_very_high")
            ctx["metrics"]["protein"] = {
                "user": user,
                "population_mean": pop_mean,
                "guideline_min": min_needed,
                "guideline_max": max_safe,
                "intermediate_target": target,
            }

        # --- Alcohol ---
        if profile.alcohol_g_day is not None and stats:
            user = profile.alcohol_g_day
            pop_mean = stats["alcohol_mean"]
            limit = self.guidelines["alcohol"]["max_g"]  # 14g/day
            target = None
            if user > limit:
                target = max(limit, (user + limit) / 2)
                ctx["notes"].append("alcohol_high")
            ctx["metrics"]["alcohol"] = {
                "user": user,
                "population_mean": pop_mean,
                "guideline_max": limit,
                "intermediate_target": target,
            }

        # --- Smoking ---
        if profile.smoking_status:
            status = profile.smoking_status.lower()
            ctx["metrics"]["smoking_status"] = {"user": status}
            if status == "current":
                ctx["notes"].append("smoker")

        return ctx
