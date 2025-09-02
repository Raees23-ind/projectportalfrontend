import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.cluster import KMeans
from django.conf import settings

MODEL_DIR = os.path.join(settings.BASE_DIR, "ml_models")
MODEL_PATH = os.path.join(MODEL_DIR, "project_pipeline.joblib")

class ProjectClusteringPipeline:
    def __init__(self, vectorizer=None, mlb=None, scaler=None, kmeans=None):
        self.vectorizer = vectorizer
        self.mlb = mlb
        self.scaler = scaler
        self.kmeans = kmeans

    def is_ready(self):
        return all([self.vectorizer is not None, self.mlb is not None, self.scaler is not None, self.kmeans is not None])

    def fit(self, df, n_clusters=3):
        """
        df must have columns: title, description, keywords, complexity_score, duration_days
        """
        # Text: combine title+description
        texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        text_vec = self.vectorizer.fit_transform(texts).toarray()

        # Tools (multi-label)
        tools_lists = df["keywords"].fillna("").apply(lambda s: [t.strip().lower() for t in s.split(",") if t.strip()]).tolist()
        self.mlb = MultiLabelBinarizer(sparse_output=False)
        tools_vec = self.mlb.fit_transform(tools_lists)

        # Numeric scaling
        numeric = df[["complexity_score", "duration_days"]].fillna(0).astype(float).values
        self.scaler = MinMaxScaler()
        numeric_scaled = self.scaler.fit_transform(numeric)

        # Concatenate
        X = np.hstack([text_vec, tools_vec, numeric_scaled])

        # Fit KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        self.kmeans.fit(X)
        return self

    def predict_single(self, data_dict):
        texts = [(data_dict.get("title", "") or "") + " " + (data_dict.get("description", "") or "")]
        text_vec = self.vectorizer.transform(texts).toarray()

        tools = [t.strip().lower() for t in (data_dict.get("keywords") or "").split(",") if t.strip()]
        tools_vec = self.mlb.transform([tools])

        numeric = np.array([[float(data_dict.get("complexity_score", 0)), float(data_dict.get("duration_days", 0))]])
        numeric_scaled = self.scaler.transform(numeric)

        X = np.hstack([text_vec, tools_vec, numeric_scaled])
        return self.kmeans.predict(X)[0]

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "mlb": self.mlb,
            "scaler": self.scaler,
            "kmeans": self.kmeans,
        }, path)

    @classmethod
    def load_if_exists(cls, path=MODEL_PATH):
        if os.path.exists(path):
            data = joblib.load(path)
            return cls(vectorizer=data["vectorizer"], mlb=data["mlb"], scaler=data["scaler"], kmeans=data["kmeans"])
        return None
