import requests
import pandas as pd
from pathlib import Path
import xport

class NHANESDownloader:
    """Download and load NHANES 2021-2023 data files"""
    
    BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"
    
    # Files we need
    FILES = {
        'demographics': 'DEMO_L.XPT',
        'body_measures': 'BMX_L.XPT',
        'blood_pressure': 'BPXO_L.XPT',
        'glucose': 'GLU_L.XPT',
        'cholesterol': 'TCHOL_L.XPT',
        'biochemistry': 'BIOPRO_L.XPT',
        'dietary': 'DR1TOT_L.XPT',
        'physical_activity': 'PAQ_L.XPT',
        'diabetes': 'DIQ_L.XPT',
        'medical_conditions': 'MCQ_L.XPT',
        'bp_questionnaire': 'BPQ_L.XPT'
    }
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, filename):
        """Download a single XPT file"""
        url = self.BASE_URL + filename
        filepath = self.data_dir / filename
        
        if filepath.exists():
            print(f"✓ {filename} already exists")
            return filepath
            
        print(f"Downloading {filename}...", end=' ')
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print("✓")
            return filepath
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def download_all(self):
        """Download all required files"""
        print("=" * 50)
        print("NHANES 2021-2023 Data Download")
        print("=" * 50)
        
        downloaded = {}
        for name, filename in self.FILES.items():
            filepath = self.download_file(filename)
            if filepath:
                downloaded[name] = filepath
        
        print(f"\n✓ Downloaded {len(downloaded)}/{len(self.FILES)} files")
        return downloaded
    
    def load_xpt(self, filepath):
        """Load XPT file into pandas DataFrame"""
        with open(filepath, 'rb') as f:
            df = xport.to_dataframe(f)
        return df
    
    def load_all_data(self):
        """Download and load all files into memory"""
        filepaths = self.download_all()
        
        print("\nLoading data into memory...")
        data = {}
        for name, filepath in filepaths.items():
            print(f"Loading {name}...", end=' ')
            try:
                data[name] = self.load_xpt(filepath)
                print(f"✓ ({len(data[name])} rows)")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        return data


# Usage
if __name__ == "__main__":
    downloader = NHANESDownloader()
    nhanes_data = downloader.load_all_data()
    
    # Quick preview
    print("\n" + "=" * 50)
    print("Data Summary")
    print("=" * 50)
    for name, df in nhanes_data.items():
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")