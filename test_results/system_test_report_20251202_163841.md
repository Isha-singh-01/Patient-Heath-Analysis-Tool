# SYSTEM TEST REPORT
Generated: 2025-12-02 16:38:41

## Summary
- Personas passed: 25/25
- Edge cases passed: 10/10
- Components passed: 5/5
- Errors logged: 0
- Composite score (0-100): 80.00

## Performance
- Avg latency (s): 7.207044410705566
- P95 latency (s): 9.330106258392334

## Component results (brief)
- model_prediction: ok=True details={'prob_sum': 0.9999999999999992}
- nhanes_profile: ok=True details={'sample_size': 1365}
- population_comparison: ok=True details={'has_glucose': True}
- lifestyle_context: ok=True details={'keys': ['disease', 'age', 'age_band', 'metrics', 'notes']}
- recommendation_generation: ok=True details={'length': 1848}

## Persona highlights (first 10)
- [1] Diabetes - Classic: pred={'disease': 'Kidney Failure', 'confidence': 0.21714437244353213, 'top_3_predictions': {'Kidney Failure': 0.21714437244353213, 'Diabetes': 0.21111150270928838, 'Heart Failure': 0.20351133352771744}, 'model_used': 'Random Forest'} stable=False avg_time=6.775s
- [2] Diabetes - Prediabetic: pred={'disease': 'Kidney Failure', 'confidence': 0.23487452722218596, 'top_3_predictions': {'Kidney Failure': 0.23487452722218596, 'Heart Failure': 0.22742976186396013, 'Diabetes': 0.13491228209743894}, 'model_used': 'Random Forest'} stable=False avg_time=5.707s
- [3] Diabetes - Type2 with complications: pred={'disease': 'Heart Failure', 'confidence': 0.21554599372858885, 'top_3_predictions': {'Heart Failure': 0.21554599372858885, 'Diabetes': 0.19767330305584038, 'Kidney Failure': 0.1969447982984177}, 'model_used': 'Random Forest'} stable=False avg_time=5.900s
- [4] Diabetes - Well-controlled: pred={'disease': 'Kidney Failure', 'confidence': 0.25393209741859346, 'top_3_predictions': {'Kidney Failure': 0.25393209741859346, 'Heart Failure': 0.24558660045304412, 'Peptic Ulcer Disease': 0.08287973172311167}, 'model_used': 'Random Forest'} stable=True avg_time=4.876s
- [5] Diabetes - Young onset: pred={'disease': 'Heart Failure', 'confidence': 0.20332084606462245, 'top_3_predictions': {'Heart Failure': 0.20332084606462245, 'Kidney Failure': 0.18805720054326547, 'Diabetes': 0.13312991326340276}, 'model_used': 'Random Forest'} stable=False avg_time=6.012s
- [6] Hypertension - Stage 2: pred={'disease': 'Hypertension', 'confidence': 0.33334288100586756, 'top_3_predictions': {'Hypertension': 0.33334288100586756, 'Heart Failure': 0.18076526876797644, 'Kidney Failure': 0.15755624992866474}, 'model_used': 'Random Forest'} stable=True avg_time=5.862s
- [7] Hypertension - Stage 1: pred={'disease': 'Hypertension', 'confidence': 0.23757594902045176, 'top_3_predictions': {'Hypertension': 0.23757594902045176, 'Heart Failure': 0.21200532889462856, 'Kidney Failure': 0.17996062325574294}, 'model_used': 'Random Forest'} stable=False avg_time=6.720s
- [8] Hypertension - Elderly: pred={'disease': 'Heart Failure', 'confidence': 0.2375660352166342, 'top_3_predictions': {'Heart Failure': 0.2375660352166342, 'Kidney Failure': 0.23507930692127688, 'Hypertension': 0.12933731658349476}, 'model_used': 'Random Forest'} stable=True avg_time=4.987s
- [9] Hypertension - With diabetes: pred={'disease': 'Hypertension', 'confidence': 0.21992309135593652, 'top_3_predictions': {'Hypertension': 0.21992309135593652, 'Heart Failure': 0.18068620595066334, 'Diabetes': 0.17138051032805712}, 'model_used': 'Random Forest'} stable=False avg_time=5.622s
- [10] Hypertension - Pre-hypertensive: pred={'disease': 'Hypertension', 'confidence': 0.2398845695329801, 'top_3_predictions': {'Hypertension': 0.2398845695329801, 'Heart Failure': 0.20654408191321125, 'Kidney Failure': 0.18138545419177057}, 'model_used': 'Random Forest'} stable=False avg_time=6.825s

## Errors (sample)