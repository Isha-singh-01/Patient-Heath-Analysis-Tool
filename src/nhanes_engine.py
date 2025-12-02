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

        # Precomputed NHANES aggregate stats and guideline thresholds
        self.nhanes_stats: List[Dict[str, Any]] = data["nhanes_stats"]
        self.guidelines: Dict[str, Any] = data["guidelines"]

    def _age_band(self, age: int) -> str:
        """
        Map numeric age to one of the bands:
        '0-17', '18-29', '30-44', '45-59', '60+'
        (Keep these EXACT as requested.)
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

    def _lookup_stats_by_age_band(self, age_band: str) -> Dict[str, Any]:
        """
        Return a NHANES summary row for this age band,
        ignoring disease labels entirely.
        """
        for row in self.nhanes_stats:
            if row.get("age_band") == age_band:
                return row
        return {}  #we'll still use guidelines + user data

    def _status_high_if_above(self, user: Optional[float], limit: Optional[float]) -> str:
        if user is None or limit is None:
            return "unknown"
        if user > limit * 1.1:
            return "high"
        if user < limit * 0.9:
            return "low"
        return "ok"

    def _status_low_if_below(self, user: Optional[float], minimum: Optional[float]) -> str:
        if user is None or minimum is None:
            return "unknown"
        if user < minimum * 0.9:
            return "low"
        if user > minimum * 1.1:
            return "high"
        return "ok"

    def build_context(self, disease: str, profile: UserProfile) -> Dict[str, Any]:
        """
        Returns a machine-friendly dict for the LLM.
        IMPORTANT: lifestyle context is primarily based on guidelines + user metrics.
        """
        band = self._age_band(profile.age)
        stats = self._lookup_stats_by_age_band(band)

        ctx: Dict[str, Any] = {
            "disease": disease,    
            "age": profile.age,
            "age_band": band,
            "metrics": {},
            "notes": [],
        }

        # convenience getter to avoid KeyError when stats is {}
        def s(key: str):
            return stats.get(key) if stats else None

        # --- Sugar (g/day) ---
        if profile.sugar_g_day is not None:
            user = profile.sugar_g_day
            pop_mean = s("sugar_mean")
            guideline = self.guidelines["sugar"]["who_limit_g"]  
            status = self._status_high_if_above(user, guideline)
            target = None
            if status == "high" and guideline is not None:
                # realistic step: halfway between user and guideline
                target = round((user + guideline) / 2, 1)
                ctx["notes"].append("sugar_high")

            ctx["metrics"]["sugar"] = {
                "user": user,
                "population_mean": pop_mean,
                "guideline": guideline,
                "intermediate_target": target,
                "unit": "g/day",
                "status": status,
            }

        # --- Sodium (mg/day) ---
        if profile.sodium_mg_day is not None:
            user = profile.sodium_mg_day
            pop_mean = s("sodium_mean")
            limit = self.guidelines["sodium"]["limit_mg"]
            status = self._status_high_if_above(user, limit)
            target = None
            if status == "high" and limit is not None:
                target = round((user + limit) / 2, 1)
                ctx["notes"].append("sodium_high")

            ctx["metrics"]["sodium"] = {
                "user": user,
                "population_mean": pop_mean,
                "guideline": limit,
                "intermediate_target": target,
                "unit": "mg/day",
                "status": status,
            }

        # --- Activity (minutes/week) ---
        if profile.activity_minutes_week is not None:
            user = profile.activity_minutes_week
            min_needed = self.guidelines["activity"]["min_minutes_week"]
            status = self._status_low_if_below(user, min_needed)
            target = None
            if status == "low" and min_needed is not None:
                target = round((user + min_needed) / 2, 1)
                ctx["notes"].append("activity_low")

            ctx["metrics"]["activity"] = {
                "user": user,
                "guideline_min": min_needed,
                "intermediate_target": target,
                "unit": "min/week",
                "status": status,
            }

        # --- BMI ---
        if profile.bmi is not None:
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

        # --- Fiber (g/day) ---
        if profile.fiber_g_day is not None:
            user = profile.fiber_g_day
            pop_mean = s("fiber_mean")
            min_needed = self.guidelines["fiber"]["min_g"]
            status = self._status_low_if_below(user, min_needed)
            target = None
            if status == "low" and min_needed is not None:
                target = round((user + min_needed) / 2, 1)
                ctx["notes"].append("fiber_low")

            ctx["metrics"]["fiber"] = {
                "user": user,
                "population_mean": pop_mean,
                "guideline_min": min_needed,
                "intermediate_target": target,
                "unit": "g/day",
                "status": status,
            }

        # --- Protein (g/day) ---
        # No hard "do not exceed" limit. We mark low / very_high and let the LLM
        # talk about 0.8–2.0 g/kg/day.
        if profile.protein_g_day is not None:
            user = profile.protein_g_day
            pop_mean = s("protein_mean")
            min_needed = self.guidelines["protein"]["min_g"]   # baseline reference
            max_ref = self.guidelines["protein"]["max_g"]      # soft reference
            status = "ok"
            if min_needed is not None and user < min_needed * 0.9:
                status = "low"
                ctx["notes"].append("protein_low")
            elif max_ref is not None and user > max_ref * 2.0:
                status = "very_high"
                ctx["notes"].append("protein_very_high")

            target = None
            if status == "low" and min_needed is not None:
                target = round((user + min_needed) / 2, 1)

            ctx["metrics"]["protein"] = {
                "user": user,
                "population_mean": pop_mean,
                "guideline_min": min_needed,
                "guideline_max_ref": max_ref,
                "intermediate_target": target,
                "unit": "g/day",
                "status": status,
            }

        # --- Alcohol (g/day) ---
        if profile.alcohol_g_day is not None:
            user = profile.alcohol_g_day
            pop_mean = s("alcohol_mean")
            limit = self.guidelines["alcohol"]["max_g"]
            status = self._status_high_if_above(user, limit)
            target = None
            if status == "high" and limit is not None:
                target = round((user + limit) / 2, 1)
                ctx["notes"].append("alcohol_high")

            ctx["metrics"]["alcohol"] = {
                "user": user,
                "population_mean": pop_mean,
                "guideline_max": limit,
                "intermediate_target": target,
                "unit": "g/day",
                "status": status,
            }

        # --- Smoking ---
        if profile.smoking_status:
            status = profile.smoking_status.lower()
            ctx["metrics"]["smoking_status"] = {"user": status}
            if status == "current":
                ctx["notes"].append("smoker")

        return ctx
