# src/model_service.py (or a Jupyter cell)

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "models"

class DiseasePredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)

        # Load model + transformers
        self.rf = joblib.load(self.model_dir / "disease_rf_model.pkl")
        self.tfidf = joblib.load(self.model_dir / "tfidf_vectorizer.pkl")
        self.ct = joblib.load(self.model_dir / "column_transformer.pkl")
        self.label_encoder = joblib.load(self.model_dir / "label_encoder.pkl")

        # Load metadata
        with open(self.model_dir / "disease_rf_metadata.json") as f:
            meta = json.load(f)

        self.text_col = meta["text_col"]
        self.cat_cols = meta["cat_cols"]
        self.num_cols = meta["num_cols"]
        self.feature_names = meta["feature_names"]
        self.target_classes = meta["target_classes"]  # these are encoded ints

        # Optional: store original disease label names for convenience
        self.class_labels = list(self.label_encoder.classes_)


    def _build_design_matrix_from_df(self, df: pd.DataFrame):
        """
        Given a DataFrame with at least text_col, cat_cols, num_cols,
        return the sparse matrix X ready for RF.predict_proba.
        """
        # Text part
        X_text = self.tfidf.transform(df[self.text_col].fillna(""))

        # Structured part
        X_struct = self.ct.transform(df[self.cat_cols + self.num_cols])

        # Combine in the same order as during training
        X = hstack([X_text, X_struct])
        return X

    def predict_proba_from_df(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_design_matrix_from_df(df)
        return self.rf.predict_proba(X)

    def predict_topk_from_df(self, df: pd.DataFrame, k: int = 3, threshold: float = 0.1):
        proba = self.predict_proba_from_df(df)[0]  # probabilities in same order as rf.classes_
        probs_dict = dict(zip(self.rf.classes_, proba))  # key: encoded int, val: prob

        # Sort by probability (descending)
        sorted_items = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)

        # Decode to original labels
        decoded_results = []
        for enc_label, p in sorted_items:
            if p < threshold:
                continue
            # enc_label is an int: decode using label encoder
            disease_name = self.label_encoder.inverse_transform([enc_label])[0]
            decoded_results.append((disease_name, float(p)))

        return decoded_results[:k]

    def predict_from_inputs(self, text: str, cats: dict, nums: dict, k: int = 3, threshold: float = 0.1):
        """
        Convenience method: build a 1-row DataFrame from raw inputs.

        text: free-text symptom description for text_col
        cats: dict for categorical columns (e.g. {'sex': 'M'})
        nums: dict for numeric columns (e.g. {'age': 55, 'bmi': 30.2})
        """
        row = {}

        # text col
        row[self.text_col] = text

        # categorical
        for col in self.cat_cols:
            row[col] = cats.get(col, None)

        # numeric
        for col in self.num_cols:
            row[col] = nums.get(col, np.nan)

        df = pd.DataFrame([row])
        return self.predict_topk_from_df(df, k=k, threshold=threshold)
