"""
system_test_runner_pro.py
Comprehensive evaluation runner (NO pytest)

Outputs:
 - system_test_full_results.json
 - system_test_summary.csv
 - system_test_personas.csv
 - system_test_report.md

Usage:
    python system_test_runner_pro.py
"""

import time
import json
import math
import statistics
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Add src to path
import sys
sys.path.append("src")

# Try to load environment & system; be resilient if missing
try:
    from load_env import load_env, get_config
    load_env()
    config = get_config()
except Exception:
    config = {
        "hf_token": os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY"),
        "hf_model": os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct:groq"),
        "use_llm": bool(os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")),
        "model_dir": "models_fixed",
        "profiles_path": "models_fixed/comprehensive_disease_profiles.json"
    }

# Import user system (must exist in src/)
try:
    from recommendation_engine import ComprehensiveRecommendationSystem
except Exception as e:
    ComprehensiveRecommendationSystem = None
    print("WARN: Could not import ComprehensiveRecommendationSystem from src/recommendation_engine.py")
    print("Exception:", e)

# ---------- Utility functions ----------
def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def entropy_from_probs(probs: List[float]) -> float:
    """Shannon entropy (nats) of probability distribution"""
    probs = np.array([p for p in probs if p > 0])
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))

def topk_from_probs(probs: List[float], k: int = 3):
    probs = np.array(probs)
    idx = np.argsort(probs)[::-1][:k]
    return idx.tolist(), probs[idx].tolist()

def safe_get(d: Dict, keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def normalized_score(val, min_val, max_val):
    if max_val == min_val:
        return 1.0
    return max(0.0, min(1.0, (val - min_val) / (max_val - min_val)))

# Heuristic hallucination detection (simple)
def hallucination_heuristic(report: Dict[str, Any], known_diseases: List[str]) -> Dict[str, Any]:
    """
    Returns flags about potential hallucinations or invented facts.
    - flags if predicted disease not in known_diseases
    - checks for absolute claims like 'cure' or improbable lab recommendations
    - checks for presence of unsupported external references
    """
    flags = {"predicted_disease_unknown": False, "mentions_unsupported_sources": False, "strong_claims": []}
    try:
        pdisease = safe_get(report, ["prediction", "disease"], "")
        if pdisease and known_diseases and pdisease not in known_diseases:
            flags["predicted_disease_unknown"] = True
    except Exception:
        pass

    # scan textual recommendations for 'cure', 'guarantee', 'always', 'never'
    txt = json.dumps(report, ensure_ascii=False).lower()
    strong_words = ["cure", "guarantee", "always", "never", "must", "definitely"]
    for w in strong_words:
        if w in txt:
            flags["strong_claims"].append(w)

    # check for external urls or citations (which could be hallucinated)
    if "http://" in txt or "https://" in txt or "www." in txt:
        flags["mentions_unsupported_sources"] = True

    return flags

# ---------- Test runner class ----------
class SystemTestRunnerPro:
    def __init__(self, model_dir: str = None, profiles_path: str = None, use_llm: bool = None, hf_api_key: str = None):
        print("\n" + "="*70)
        print("INITIALIZING Comprehensive System Test Runner (Pro)")
        print("="*70)

        # config resolution
        self.model_dir = model_dir or config.get("model_dir")
        self.profiles_path = profiles_path or config.get("profiles_path")
        self.use_llm = use_llm if use_llm is not None else config.get("use_llm", False)
        self.hf_api_key = hf_api_key or config.get("hf_token")

        self.system = None
        self.known_diseases = []
        self.component_status = {}
        self.results = {
            "metadata": {
                "started_at": datetime.now().isoformat(),
                "model_dir": self.model_dir,
                "profiles_path": self.profiles_path,
                "use_llm": self.use_llm
            },
            "performance": {},
            "edge_cases": [],
            "component_tests": [],
            "personas": [],
            "errors": [],
            "raw_logs": []
        }

        # try to instantiate system
        if ComprehensiveRecommendationSystem is None:
            err = "ComprehensiveRecommendationSystem class not found; import failed."
            print("ERROR:", err)
            self.results["errors"].append({"phase": "init", "error": err})
            return

        try:
            self.system = ComprehensiveRecommendationSystem(
                model_dir=self.model_dir,
                profiles_path=self.profiles_path,
                use_llm=self.use_llm,
                hf_api_key=self.hf_api_key
            )
            # attempt loading list of diseases if available
            try:
                self.known_diseases = getattr(self.system, "disease_names", []) or []
            except Exception:
                self.known_diseases = []
            print("✅ System instantiated")
            self.component_status["system_instantiated"] = True
        except Exception as e:
            self.component_status["system_instantiated"] = False
            msg = f"Failed to instantiate system: {e}"
            print("ERROR:", msg)
            self.results["errors"].append({"phase": "init", "error": msg})
            return

    # ---------------- Performance test ----------------
    def test_performance(self, n_iterations: int = 10, input_sample: Dict[str, Any] = None, target_seconds: float = 2.0):
        print("\n" + "="*70)
        print("TEST: PERFORMANCE")
        print("="*70)
        if self.system is None:
            msg = "System not available for performance test"
            print("ERROR:", msg)
            self.results["errors"].append({"phase": "performance", "error": msg})
            return

        sample = input_sample or {
            "age": 55, "gender": "M",
            "symptoms_text": "chest pain shortness of breath fatigue",
            "glucose": 120, "systolic_bp": 140, "bmi": 28.5
        }

        times = []
        success_count = 0
        failures = []

        for i in range(n_iterations):
            t0 = time.time()
            try:
                report = self.system.generate_comprehensive_report(sample)
                elapsed = time.time() - t0
                times.append(elapsed)
                success_count += 1
                # capture minimal sanity info
                self.results["raw_logs"].append({
                    "test": "performance_run",
                    "iteration": i + 1,
                    "elapsed": elapsed,
                    "prediction": safe_get(report, ["prediction", "disease"], None)
                })
                print(f"  run {i+1}/{n_iterations}: {elapsed:.3f}s -> {safe_get(report, ['prediction','disease'], 'N/A')}")
            except Exception as e:
                elapsed = time.time() - t0
                failures.append({"iteration": i + 1, "error": str(e)})
                self.results["raw_logs"].append({
                    "test": "performance_run",
                    "iteration": i + 1,
                    "elapsed": elapsed,
                    "error": str(e)
                })
                print(f"  run {i+1}/{n_iterations}: FAILED -> {e}")

        if times:
            stats = {
                "iterations": n_iterations,
                "successful": success_count,
                "failed": len(failures),
                "avg_s": float(np.mean(times)),
                "median_s": float(np.median(times)),
                "min_s": float(np.min(times)),
                "max_s": float(np.max(times)),
                "std_s": float(np.std(times)),
                "p50_s": float(np.percentile(times, 50)),
                "p90_s": float(np.percentile(times, 90)),
                "p95_s": float(np.percentile(times, 95)),
                "p99_s": float(np.percentile(times, 99)),
                "target_seconds": target_seconds,
                "passed": float(np.mean(times)) < target_seconds
            }
            self.results["performance"] = stats
            print("Performance summary:", stats)
        else:
            self.results["performance"] = {"iterations": n_iterations, "successful": 0, "failed": len(failures)}

        if failures:
            self.results["errors"].append({"phase": "performance", "failures": failures})

    # ---------------- Edge cases ----------------
    def test_edge_cases(self):
        print("\n" + "="*70)
        print("TEST: EDGE CASES")
        print("="*70)

        cases = [
            ("Missing all data", {"age": 45, "gender": "M", "symptoms_text": ""}),
            ("Only symptoms", {"age": 60, "gender": "F", "symptoms_text": "headache dizziness"}),
            ("Extreme glucose high", {"age": 55, "gender": "M", "symptoms_text": "confusion shakiness", "glucose": 500}),
            ("Extreme glucose low", {"age": 55, "gender": "M", "symptoms_text": "weakness shakiness", "glucose": 30}),
            ("Very high BP", {"age": 70, "gender": "M", "symptoms_text": "severe headache", "systolic_bp": 220, "diastolic_bp": 130}),
            ("High creatinine", {"age": 65, "gender": "F", "symptoms_text": "fatigue nausea", "creatinine": 8.5}),
            ("Unclear symptoms", {"age": 40, "gender": "M", "symptoms_text": "just not feeling well"}),
            ("Multiple mixed", {"age": 58, "gender": "F", "symptoms_text": "chest pain fatigue dizziness shortness of breath nausea headache", "glucose": 180, "systolic_bp": 160, "creatinine": 2.1}),
            ("Young adult", {"age": 18, "gender": "M", "symptoms_text": "chest pain rapid heartbeat", "systolic_bp": 140}),
            ("Very elderly", {"age": 95, "gender": "F", "symptoms_text": "fatigue confusion weakness", "glucose": 95, "creatinine": 1.8})
        ]

        for idx, (name, inp) in enumerate(cases, start=1):
            rec = {"case_name": name, "input": inp, "passed": False, "response_time_s": None, "prediction": None, "top3": None, "entropy": None, "hallucination_flags": None, "error": None}
            t0 = time.time()
            try:
                report = self.system.generate_comprehensive_report(inp)
                elapsed = time.time() - t0
                rec["response_time_s"] = elapsed
                rec["prediction"] = safe_get(report, ["prediction", "disease"], None)
                rec["top3"] = safe_get(report, ["prediction", "top_3_predictions"], None)
                # if probabilities accessible, compute entropy
                probs = safe_get(report, ["prediction", "probabilities"], None)
                if probs and isinstance(probs, (list, tuple)):
                    rec["entropy"] = entropy_from_probs(probs)
                else:
                    rec["entropy"] = None

                # hallucination heuristic
                rec["hallucination_flags"] = hallucination_heuristic(report, self.known_diseases)

                # basic structural checks
                ok = True
                if not rec["prediction"]:
                    ok = False
                if not safe_get(report, ["recommendations"], None):
                    ok = False

                rec["passed"] = bool(ok)
                print(f"[{idx}/{len(cases)}] {name}: passed={rec['passed']} time={elapsed:.3f}s pred={rec['prediction']}")
            except Exception as e:
                rec["error"] = str(e)
                rec["response_time_s"] = time.time() - t0
                print(f"[{idx}/{len(cases)}] {name}: FAILED -> {e}")
                self.results["errors"].append({"phase": "edge_case", "case": name, "error": str(e)})
            self.results["edge_cases"].append(rec)

    # ---------------- Component integration ----------------
    def test_components(self):
        print("\n" + "="*70)
        print("TEST: COMPONENT INTEGRATION")
        print("="*70)

        comp_results = []
        # 1. Feature extraction + RF predict_proba
        comp_name = "model_prediction"
        try:
            sample = {"age": 55, "gender": "M", "symptoms_text": "frequent urination increased thirst", "glucose": 145, "bmi": 31}
            features = self.system.prepare_input_features(sample)
            probs = None
            try:
                probs = self.system.rf_model.predict_proba(features)[0].tolist()
            except Exception as e:
                # maybe rf_model has predict_proba on different input shape
                try:
                    probs = self.system.rf_model.predict_proba(features.values.reshape(1, -1))[0].tolist()
                except Exception:
                    raise e
            ok = bool(probs and sum(probs) > 0)
            comp_results.append({"component": comp_name, "ok": ok, "details": {"prob_sum": float(sum(probs)) if probs else None}})
            print(f"  {comp_name}: ok={ok}")
        except Exception as e:
            comp_results.append({"component": comp_name, "ok": False, "details": {"error": str(e)}})
            print(f"  {comp_name}: ERROR -> {e}")
            self.results["errors"].append({"phase": "component", "component": comp_name, "error": str(e)})

        # 2. NHANES profile retrieval
        comp_name = "nhanes_profile"
        try:
            profile = self.system.get_nhanes_profile("Diabetes")
            ok = bool(profile and profile.get("sample_size", 0) > 0)
            comp_results.append({"component": comp_name, "ok": ok, "details": {"sample_size": profile.get("sample_size", None)}})
            print(f"  {comp_name}: ok={ok}")
        except Exception as e:
            comp_results.append({"component": comp_name, "ok": False, "details": {"error": str(e)}})
            print(f"  {comp_name}: ERROR -> {e}")
            self.results["errors"].append({"phase": "component", "component": comp_name, "error": str(e)})

        # 3. Population comparison
        comp_name = "population_comparison"
        try:
            sample = {"age": 55, "gender": "M", "symptoms_text": "", "glucose": 145}
            comparisons = self.system.compare_user_to_population(sample, profile)
            ok = bool("glucose" in comparisons and "status" in comparisons["glucose"])
            comp_results.append({"component": comp_name, "ok": ok, "details": {"has_glucose": ok}})
            print(f"  {comp_name}: ok={ok}")
        except Exception as e:
            comp_results.append({"component": comp_name, "ok": False, "details": {"error": str(e)}})
            print(f"  {comp_name}: ERROR -> {e}")
            self.results["errors"].append({"phase": "component", "component": comp_name, "error": str(e)})

        # 4. Lifestyle context
        comp_name = "lifestyle_context"
        try:
            from nhanes_lifestyle_engine import UserMetrics
            metrics = UserMetrics(age=55, glucose=145, bmi=31, sugar_g_day=80, activity_minutes_week=30)
            context = self.system.lifestyle_recommender.build_lifestyle_context("Diabetes", metrics)
            ok = bool("metrics" in context and "notes" in context)
            comp_results.append({"component": comp_name, "ok": ok, "details": {"keys": list(context.keys())}})
            print(f"  {comp_name}: ok={ok}")
        except Exception as e:
            comp_results.append({"component": comp_name, "ok": False, "details": {"error": str(e)}})
            print(f"  {comp_name}: ERROR -> {e}")
            self.results["errors"].append({"phase": "component", "component": comp_name, "error": str(e)})

        # 5. Recommendation generation (fallback)
        comp_name = "recommendation_generation"
        try:
            # try to produce fallback recommendations using system helper
            probs = {"Diabetes": 0.85, "Hypertension": 0.10, "Heart Failure": 0.05}
            recs_text = self.system._generate_fallback_recommendations("Diabetes", profile, comparisons, probs, context, sample)
            ok = isinstance(recs_text, str) and len(recs_text) > 100
            comp_results.append({"component": comp_name, "ok": ok, "details": {"length": len(recs_text) if recs_text else 0}})
            print(f"  {comp_name}: ok={ok}")
        except Exception as e:
            comp_results.append({"component": comp_name, "ok": False, "details": {"error": str(e)}})
            print(f"  {comp_name}: ERROR -> {e}")
            self.results["errors"].append({"phase": "component", "component": comp_name, "error": str(e)})

        self.results["component_tests"] = comp_results

    # ---------------- Personas ----------------
    def _generate_25_personas(self) -> List[Dict[str, Any]]:
        """Return the 25 detailed personas you provided earlier (expanded)."""
        # Keep the exact personas described in your earlier long list (condensed and explicit)
        personas = [
            # 5 Diabetes
            {"name": "Diabetes - Classic", "age":55, "gender":"M", "risk_category":"High", "expected":["Diabetes"], "input":{
                "age":55,"gender":"M","bmi":32.5,"symptoms_text":"increased thirst frequent urination fatigue blurred vision","glucose":185,"systolic_bp":142,"sugar_g_day":120,"activity_minutes_week":20,"fiber_g_day":10
            }},
            {"name":"Diabetes - Prediabetic","age":48,"gender":"F","risk_category":"Medium","expected":["Prediabetes"],"input":{
                "age":48,"gender":"F","bmi":27.5,"symptoms_text":"fatigue increased hunger","glucose":115,"systolic_bp":128,"sugar_g_day":85,"activity_minutes_week":60
            }},
            {"name":"Diabetes - Type2 with complications","age":62,"gender":"M","risk_category":"High","expected":["Diabetes"],"input":{
                "age":62,"gender":"M","bmi":35.0,"symptoms_text":"numbness in feet blurred vision frequent urination","glucose":220,"creatinine":1.4,"systolic_bp":155,"sugar_g_day":140,"activity_minutes_week":0
            }},
            {"name":"Diabetes - Well-controlled","age":58,"gender":"F","risk_category":"Low","expected":["Diabetes"],"input":{
                "age":58,"gender":"F","bmi":24.5,"symptoms_text":"mild fatigue","glucose":105,"systolic_bp":122,"sugar_g_day":40,"activity_minutes_week":180,"fiber_g_day":30
            }},
            {"name":"Diabetes - Young onset","age":32,"gender":"M","risk_category":"Medium","expected":["Diabetes"],"input":{
                "age":32,"gender":"M","bmi":29.0,"symptoms_text":"increased thirst weight loss","glucose":165,"sugar_g_day":95,"activity_minutes_week":45
            }},

            # 5 Hypertension
            {"name":"Hypertension - Stage 2","age":64,"gender":"M","risk_category":"High","expected":["Hypertension"],"input":{
                "age":64,"gender":"M","bmi":30.0,"symptoms_text":"headache dizziness chest discomfort","systolic_bp":165,"diastolic_bp":102,"sodium_mg_day":4200,"activity_minutes_week":30
            }},
            {"name":"Hypertension - Stage 1","age":52,"gender":"F","risk_category":"Medium","input":{
                "age":52,"gender":"F","bmi":27.8,"symptoms_text":"occasional headaches","systolic_bp":138,"diastolic_bp":88,"sodium_mg_day":3200,"activity_minutes_week":90
            }},
            {"name":"Hypertension - Elderly","age":78,"gender":"F","risk_category":"High","input":{
                "age":78,"gender":"F","bmi":26.0,"symptoms_text":"dizziness fatigue","systolic_bp":158,"diastolic_bp":92,"sodium_mg_day":3500,"activity_minutes_week":50
            }},
            {"name":"Hypertension - With diabetes","age":60,"gender":"M","risk_category":"High","input":{
                "age":60,"gender":"M","bmi":33.0,"symptoms_text":"headache blurred vision frequent urination","glucose":156,"systolic_bp":148,"diastolic_bp":94,"sodium_mg_day":3800,"sugar_g_day":100
            }},
            {"name":"Hypertension - Pre-hypertensive","age":45,"gender":"M","risk_category":"Low","input":{
                "age":45,"gender":"M","bmi":26.5,"symptoms_text":"mild headaches stress","systolic_bp":128,"diastolic_bp":82,"sodium_mg_day":2800,"activity_minutes_week":120
            }},

            # 5 Heart Failure
            {"name":"Heart Failure - Severe","age":72,"gender":"M","risk_category":"High","expected":["Heart Failure"],"input":{
                "age":72,"gender":"M","bmi":28.0,"symptoms_text":"severe shortness of breath swelling in legs fatigue","systolic_bp":145,"creatinine":1.6,"sodium_mg_day":4000,"activity_minutes_week":10
            }},
            {"name":"Heart Failure - Compensated","age":68,"gender":"F","risk_category":"Medium","input":{
                "age":68,"gender":"F","bmi":25.5,"symptoms_text":"shortness of breath on exertion mild fatigue","systolic_bp":132,"creatinine":1.1,"sodium_mg_day":2100,"activity_minutes_week":75
            }},
            {"name":"Heart Failure - With AFib","age":75,"gender":"M","risk_category":"High","input":{
                "age":75,"gender":"M","bmi":27.0,"symptoms_text":"irregular heartbeat shortness of breath fatigue","systolic_bp":138,"potassium":3.8,"activity_minutes_week":40
            }},
            {"name":"Heart Failure - Post-MI","age":66,"gender":"M","risk_category":"High","input":{
                "age":66,"gender":"M","bmi":29.5,"symptoms_text":"chest pain shortness of breath swelling","systolic_bp":142,"glucose":128,"sodium_mg_day":3400,"activity_minutes_week":25
            }},
            {"name":"Heart Failure - Young presentation","age":52,"gender":"M","risk_category":"Medium","input":{
                "age":52,"gender":"M","bmi":31.0,"symptoms_text":"shortness of breath fatigue chest discomfort","systolic_bp":135,"activity_minutes_week":45
            }},

            # 5 Kidney Failure
            {"name":"Kidney Failure - Advanced CKD","age":70,"gender":"M","risk_category":"High","expected":["Kidney Failure"],"input":{
                "age":70,"gender":"M","bmi":28.0,"symptoms_text":"fatigue nausea decreased urine output swelling","creatinine":3.8,"urea_nitrogen":65,"potassium":5.5,"protein_g_day":120,"sodium_mg_day":4500
            }},
            {"name":"Kidney Failure - Stage 3 CKD","age":65,"gender":"F","risk_category":"Medium","input":{
                "age":65,"gender":"F","bmi":26.5,"symptoms_text":"fatigue mild swelling","creatinine":1.8,"urea_nitrogen":28,"protein_g_day":85,"sodium_mg_day":2800
            }},
            {"name":"Kidney Failure - Diabetic nephropathy","age":62,"gender":"M","risk_category":"High","input":{
                "age":62,"gender":"M","bmi":32.0,"symptoms_text":"fatigue increased urination swelling feet","glucose":195,"creatinine":2.4,"systolic_bp":152,"sugar_g_day":110,"protein_g_day":105
            }},
            {"name":"Kidney Failure - Hypertensive nephropathy","age":68,"gender":"F","risk_category":"High","input":{
                "age":68,"gender":"F","bmi":29.0,"symptoms_text":"headache fatigue decreased urine","systolic_bp":168,"creatinine":2.1,"urea_nitrogen":42,"sodium_mg_day":4100
            }},
            {"name":"Kidney Failure - Early CKD","age":58,"gender":"M","risk_category":"Low","input":{
                "age":58,"gender":"M","bmi":27.5,"symptoms_text":"mild fatigue","creatinine":1.3,"glucose":108,"protein_g_day":95,"activity_minutes_week":100
            }},

            # 5 Mixed/Other
            {"name":"Asthma - Moderate","age":42,"gender":"F","risk_category":"Medium","input":{
                "age":42,"gender":"F","bmi":24.0,"symptoms_text":"wheezing shortness of breath chest tightness","activity_minutes_week":110
            }},
            {"name":"Healthy - Active","age":35,"gender":"M","risk_category":"Low","input":{
                "age":35,"gender":"M","bmi":23.5,"symptoms_text":"occasional fatigue","glucose":92,"systolic_bp":115,"sugar_g_day":35,"activity_minutes_week":240,"fiber_g_day":32
            }},
            {"name":"Multiple conditions - Elderly","age":82,"gender":"F","risk_category":"High","input":{
                "age":82,"gender":"F","bmi":24.0,"symptoms_text":"fatigue dizziness confusion weakness","glucose":142,"systolic_bp":158,"creatinine":1.7,"activity_minutes_week":30
            }},
            {"name":"Obesity - Sedentary","age":44,"gender":"M","risk_category":"High","input":{
                "age":44,"gender":"M","bmi":38.5,"symptoms_text":"joint pain fatigue shortness of breath","glucose":118,"systolic_bp":136,"sugar_g_day":135,"activity_minutes_week":5,"fiber_g_day":8
            }},
            {"name":"UTI - Uncomplicated","age":38,"gender":"F","risk_category":"Low","input":{
                "age":38,"gender":"F","bmi":22.5,"symptoms_text":"burning urination frequent urination","glucose":95,"activity_minutes_week":150
            }},
        ]
        return personas

    def run_persona_tests(self):
        print("\n" + "="*70)
        print("TEST: PERSONA SUITE (25 personas)")
        print("="*70)

        personas = self._generate_25_personas()
        persona_results = []
        # For stability: run each persona twice and compare predictions
        for idx, persona in enumerate(personas, start=1):
            rec = {
                "index": idx,
                "name": persona.get("name"),
                "input": persona.get("input"),
                "expected": persona.get("expected", []),
                "predictions": [],
                "top3_probs": [],
                "entropy": None,
                "response_times": [],
                "stability_same_top1": None,
                "hallucination_flags": None,
                "passed": False,
                "error": None
            }
            try:
                # Run twice
                for run_no in (1, 2):
                    t0 = time.time()
                    r = self.system.generate_comprehensive_report(persona["input"])
                    elapsed = time.time() - t0
                    pred = safe_get(r, ["prediction", "disease"], None)
                    confidence = safe_get(r, ["prediction", "confidence"], None) or safe_get(r, ["prediction", "probabilities", 0], None)
                    top3 = safe_get(r, ["prediction", "top_3_predictions"], None)
                    probs = safe_get(r, ["prediction", "probabilities"], None)
                    rec["predictions"].append(pred)
                    rec["top3_probs"].append({"top3": top3, "probs": probs})
                    rec["response_times"].append(elapsed)
                    if probs and isinstance(probs, (list, tuple)):
                        rec["entropy"] = entropy_from_probs(probs)
                    # small delay not required
                # stability
                rec["stability_same_top1"] = (rec["predictions"][0] == rec["predictions"][1])
                # hallucination heuristic on the last report
                final_report = self.system.generate_comprehensive_report(persona["input"])
                rec["hallucination_flags"] = hallucination_heuristic(final_report, self.known_diseases)
                # decide pass criteria:
                # - returned a prediction
                # - produced recommendations
                try:
                    ok1 = bool(rec["predictions"][0])
                    ok2 = bool(safe_get(final_report, ["recommendations"], None))
                    rec["passed"] = bool(ok1 and ok2)
                except Exception:
                    rec["passed"] = False
                print(f"[{idx}/25] {rec['name']}: pred={rec['predictions'][0]} stable={rec['stability_same_top1']} time={np.mean(rec['response_times']):.3f}s passed={rec['passed']}")
            except Exception as e:
                rec["error"] = str(e)
                rec["passed"] = False
                print(f"[{idx}/25] {rec['name']}: ERROR -> {e}")
                self.results["errors"].append({"phase": "persona", "name": persona.get("name"), "error": str(e)})
            persona_results.append(rec)
        self.results["personas"] = persona_results

    # ---------------- Aggregation & scoring ----------------
    def aggregate_and_score(self):
        print("\n" + "="*70)
        print("AGGREGATING RESULTS & COMPUTING SCORES")
        print("="*70)
        # Basic tallies
        persona_count = len(self.results.get("personas", []))
        persona_passed = sum(1 for p in self.results.get("personas", []) if p.get("passed"))
        edge_count = len(self.results.get("edge_cases", []))
        edge_passed = sum(1 for e in self.results.get("edge_cases", []) if e.get("passed"))
        component_count = len(self.results.get("component_tests", []))
        component_passed = sum(1 for c in self.results.get("component_tests", []) if c.get("ok"))

        # Compose summary table rows
        summary_rows = []
        # Performance row
        perf = self.results.get("performance", {})
        summary_rows.append({
            "metric": "performance_avg_s",
            "value": perf.get("avg_s")
        })
        summary_rows.append({
            "metric": "performance_p95_s",
            "value": perf.get("p95_s")
        })
        # System-level pass rates
        summary_rows.append({"metric": "personas_passed", "value": f"{persona_passed}/{persona_count}"})
        summary_rows.append({"metric": "edge_cases_passed", "value": f"{edge_passed}/{edge_count}"})
        summary_rows.append({"metric": "components_passed", "value": f"{component_passed}/{component_count}"})
        summary_rows.append({"metric": "total_errors", "value": len(self.results.get("errors", []))})

        # Composite score: weight personas 40%, edge 20%, components 20%, perf 20%
        w_persona = 0.4
        w_edge = 0.2
        w_comp = 0.2
        w_perf = 0.2
        persona_score = (persona_passed / persona_count) if persona_count else 0.0
        edge_score = (edge_passed / edge_count) if edge_count else 0.0
        comp_score = (component_passed / component_count) if component_count else 0.0

        # performance normalized: target is mean < 2.0s and p95 < 4.0s
        perf_mean = perf.get("avg_s", None)
        perf_p95 = perf.get("p95_s", None)
        perf_score = 0.0
        if perf_mean is not None and perf_p95 is not None:
            # Map perf_mean from [0,5] seconds to [1,0] (lower is better)
            pm = normalized_score(5 - perf_mean, 0, 5)
            pp = normalized_score(5 - perf_p95, 0, 5)
            perf_score = float((pm * 0.6) + (pp * 0.4))
        composite = float(100 * (w_persona * persona_score + w_edge * edge_score + w_comp * comp_score + w_perf * perf_score))
        self.results["summary"] = {
            "persona_count": persona_count,
            "persona_passed": persona_passed,
            "edge_count": edge_count,
            "edge_passed": edge_passed,
            "component_count": component_count,
            "component_passed": component_passed,
            "errors": len(self.results.get("errors", [])),
            "composite_score_0_100": composite
        }
        # summary rows stored
        self.results["summary_rows"] = summary_rows
        print("Composite score (0-100):", composite)

    # ---------------- Save outputs ----------------
    def save_outputs(self, out_dir: str = "test_results_pro"):
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Full JSON (detailed)
        json_file = outp / f"system_test_full_results_{ts}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print("Saved full JSON:", json_file)

        # Summary CSV
        summary_rows = self.results.get("summary_rows", [])
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            csv_file = outp / f"system_test_summary_{ts}.csv"
            df_summary.to_csv(csv_file, index=False, encoding="utf-8")
            print("Saved summary CSV:", csv_file)

        # Personas CSV
        persona_rows = []
        for p in self.results.get("personas", []):
            persona_rows.append({
                "index": p.get("index"),
                "name": p.get("name"),
                "pred_top1_run1": p.get("predictions", [None])[0] if p.get("predictions") else None,
                "pred_top1_run2": p.get("predictions", [None, None])[1] if p.get("predictions") and len(p.get("predictions"))>1 else None,
                "stability_same_top1": p.get("stability_same_top1"),
                "entropy": p.get("entropy"),
                "mean_response_time_s": float(np.mean(p.get("response_times"))) if p.get("response_times") else None,
                "passed": p.get("passed"),
                "hallucination_pred_unknown": p.get("hallucination_flags", {}).get("predicted_disease_unknown") if p.get("hallucination_flags") else None
            })
        if persona_rows:
            df_personas = pd.DataFrame(persona_rows)
            personas_csv = outp / f"system_test_personas_{ts}.csv"
            df_personas.to_csv(personas_csv, index=False, encoding="utf-8")
            print("Saved personas CSV:", personas_csv)

        # human readable markdown report
        md_lines = []
        md_lines.append("# SYSTEM TEST REPORT")
        md_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("")
        md_lines.append("## Summary")
        s = self.results.get("summary", {})
        md_lines.append(f"- Personas passed: {s.get('persona_passed')}/{s.get('persona_count')}")
        md_lines.append(f"- Edge cases passed: {s.get('edge_passed')}/{s.get('edge_count')}")
        md_lines.append(f"- Components passed: {s.get('component_passed')}/{s.get('component_count')}")
        md_lines.append(f"- Errors logged: {s.get('errors')}")
        md_lines.append(f"- Composite score (0-100): {s.get('composite_score_0_100'):.2f}")
        md_lines.append("")
        md_lines.append("## Performance")
        perf = self.results.get("performance", {})
        md_lines.append(f"- Avg latency (s): {perf.get('avg_s')}")
        md_lines.append(f"- P95 latency (s): {perf.get('p95_s')}")
        md_lines.append("")
        md_lines.append("## Component results (brief)")
        for c in self.results.get("component_tests", []):
            md_lines.append(f"- {c.get('component')}: ok={c.get('ok')} details={c.get('details')}")
        md_lines.append("")
        md_lines.append("## Persona highlights (first 10)")
        for p in self.results.get("personas", [])[:10]:
            md_lines.append(f"- [{p.get('index')}] {p.get('name')}: pred={p.get('predictions')[0] if p.get('predictions') else None} stable={p.get('stability_same_top1')} avg_time={np.mean(p.get('response_times')):.3f}s" if p.get('response_times') else f"- [{p.get('index')}] {p.get('name')}: ERROR")
        md_lines.append("")
        md_lines.append("## Errors (sample)")
        for e in self.results.get("errors", [])[:20]:
            md_lines.append(f"- {e}")
        md_text = "\n".join(md_lines)
        md_file = outp / f"system_test_report_{ts}.md"
        safe_write_text(md_file, md_text)
        print("Saved markdown report:", md_file)

        print("\nAll outputs saved to:", outp.resolve())

    # ---------------- Run all ----------------
    def run_all(self):
        if self.system is None:
            print("System not initialized. Aborting run.")
            return
        self.test_performance(n_iterations=5)
        self.test_edge_cases()
        self.test_components()
        self.run_persona_tests()
        self.aggregate_and_score()
        self.save_outputs()

# ---------- run if script ----------
if __name__ == "__main__":
    runner = SystemTestRunnerPro()
    runner.run_all()
