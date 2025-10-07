import pandas as pd

# -------------------------------
# STEP 1: Read all files
# -------------------------------
print("Reading data...")

# Paths
patients_path = "data/patients.csv.gz"
diagnoses_path = "data/diagnoses_icd.csv.gz"
icd_dict_path = "data/d_icd_diagnoses.csv.gz"
labs_dict_path = "data/d_labitems.csv.gz"   # optional if available

# Read compressed CSVs directly
patients = pd.read_csv(patients_path, compression='gzip')
diagnoses = pd.read_csv(diagnoses_path, compression='gzip')
icd_dict = pd.read_csv(icd_dict_path, compression='gzip')
lab_dict = pd.read_csv(labs_dict_path, compression='gzip')

print(f"Patients: {patients.shape}")
print(f"Diagnoses: {diagnoses.shape}")
print(f"ICD Dict: {icd_dict.shape}")
print(f"Lab Dict: {lab_dict.shape}")

# -------------------------------
# STEP 2: Merge disease info
# -------------------------------
diagnoses_full = diagnoses.merge(icd_dict, on=["icd_code", "icd_version"], how="left")

# keep only first diagnosis per admission (primary diagnosis)
if "seq_num" in diagnoses_full.columns:
    diagnoses_full = diagnoses_full[diagnoses_full["seq_num"] == 1]

# -------------------------------
# STEP 3: Merge with patient demographics
# -------------------------------
patient_diagnosis = diagnoses_full.merge(patients, on="subject_id", how="left")

# Select key columns
patient_diagnosis = patient_diagnosis[[
    "subject_id",
    "hadm_id",
    "gender",
    "anchor_age",
    "icd_code",
    "icd_version",
    "long_title"
]]

print("Merged patient + diagnosis data:")
print(patient_diagnosis.head())

# -------------------------------
# STEP 4: Save intermediate file
# -------------------------------
patient_diagnosis.to_csv("patient_disease_summary.csv", index=False)
print("✅ Saved: patient_disease_summary.csv")

# -------------------------------
# STEP 5 (Next Step): Extract lab values like Glucose, Creatinine, Hemoglobin, Cholesterol
# -------------------------------
# You’ll need `labevents.csv.gz` for this — it’s very large (tens of GBs),
# so filter only for these key lab tests before loading full data.

print("\n💡 Next step: When you download labevents.csv.gz, we’ll filter key biomarkers like:")
print("   ['Glucose', 'Creatinine', 'Hemoglobin', 'Cholesterol'] using d_labitems.csv.gz")
