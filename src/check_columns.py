from data_loader_nhanes import NHANESDownloader

downloader = NHANESDownloader()
raw_data = downloader.load_all_data()

datasets_to_check = [
    'demographics',
    'body_measures', 
    'blood_pressure',
    'glucose',
    'cholesterol',
    'biochemistry',
    'dietary',
    'physical_activity',
    'diabetes',
    'medical_conditions',
    'bp_questionnaire'
]

print("\n" + "="*70)
print("NHANES 2021-2023 COLUMN NAMES FOR ALL DATASETS")
print("="*70)

for dataset_name in datasets_to_check:
    print(f"\n{dataset_name.upper().replace('_', ' ')}:")
    print("-" * 70)
    cols = raw_data[dataset_name].columns.tolist()
    print(f"Total columns: {len(cols)}")
    print(f"Columns: {cols}")
    print()