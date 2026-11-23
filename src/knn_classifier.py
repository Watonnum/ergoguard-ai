"""
Module 2: src/knn_classifier.py
K-Nearest Neighbors (K-NN) Posture Classifier using Euclidean Distance
and StandardScaler Pipeline for CSC472 Artificial Intelligence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from src.data_generator import DEFAULT_CSV_PATH, POSTURE_CLASSES, ensure_dataset_exists

FEATURE_COLUMNS = ["neck_angle", "torso_angle", "shoulder_tilt"]

LABEL_THAI_MAP = {
    "Good Posture": "ท่านั่งปกติ (Good Posture)",
    "Forward Head": "อาการคอยื่น (Forward Head / Tech Neck)",
    "Slouching": "อาการนั่งหลังค่อม / โน้มตัว (Slouching)",
}


class ErgonomicFeatureWeighter(BaseEstimator, TransformerMixin):
    """
    After StandardScaler normalizes all features to unit variance, weight
    neck_angle (1.0) and torso_angle (1.0) as primary ergonomic posture axes
    and shoulder_tilt (0.20) as a secondary alignment axis so camera perspective
    shoulder tilt never overrides neck/torso posture in Euclidean distance.
    """

    def __init__(self, weights: tuple[float, float, float] = (1.0, 1.0, 0.20)) -> None:
        self.weights = weights

    def fit(self, X: np.ndarray, y: Any = None) -> "ErgonomicFeatureWeighter":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        w = np.asarray(self.weights, dtype=float)
        return X * w


class PostureKNNClassifier:
    """
    K-NN Posture Classifier wrapped in a scikit-learn Pipeline with StandardScaler.
    Classifies upper-body posture into:
      - 'Good Posture'
      - 'Forward Head'
      - 'Slouching'
    """

    def __init__(self, csv_path: str | Path = DEFAULT_CSV_PATH, n_neighbors: int = 5) -> None:
        self.csv_path = Path(csv_path)
        self.n_neighbors = int(n_neighbors)
        self.pipeline: Pipeline = self._build_pipeline(self.n_neighbors)
        self.dataset_df: pd.DataFrame = pd.DataFrame()
        self.cv_accuracy: float = 0.0
        self.train(self.csv_path)

    @staticmethod
    def _build_pipeline(n_neighbors: int) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("weighter", ErgonomicFeatureWeighter()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        metric="euclidean",
                        weights="distance",
                    ),
                ),
            ]
        )

    def set_k(self, n_neighbors: int) -> None:
        """Update K value and retrain if needed."""
        n_neighbors = int(n_neighbors)
        if n_neighbors != self.n_neighbors:
            self.n_neighbors = n_neighbors
            self.train(self.csv_path)

    def train(self, csv_path: str | Path | None = None) -> float:
        """
        Load dataset from csv_path (generating seed data if missing) and fit
        StandardScaler + KNeighborsClassifier.
        Returns 5-fold cross-validation accuracy.
        """
        if csv_path is not None:
            self.csv_path = Path(csv_path)

        ensure_dataset_exists(self.csv_path)
        df = pd.read_csv(self.csv_path)

        for col in FEATURE_COLUMNS + ["label"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {self.csv_path}")

        self.dataset_df = df.copy()
        X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = df["label"].astype(str).to_numpy()

        effective_k = min(self.n_neighbors, len(df))
        self.pipeline = self._build_pipeline(effective_k)
        self.pipeline.fit(X, y)

        if len(df) >= 15:
            scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="accuracy")
            self.cv_accuracy = float(np.mean(scores))
        else:
            self.cv_accuracy = float(self.pipeline.score(X, y))

        return self.cv_accuracy

    def _transform_query(self, query: np.ndarray) -> np.ndarray:
        scaler: StandardScaler = self.pipeline.named_steps["scaler"]
        weighter: ErgonomicFeatureWeighter = self.pipeline.named_steps["weighter"]
        return weighter.transform(scaler.transform(query))

    def predict(
        self,
        neck_angle: float,
        torso_angle: float,
        shoulder_tilt: float,
    ) -> tuple[str, float, list[float]]:
        """
        Predict posture class for the given upper-body angles.

        Returns:
            (predicted_label, confidence_score, nearest_distances)
        """
        query = np.array([[float(neck_angle), float(torso_angle), float(shoulder_tilt)]], dtype=float)
        predicted_label = str(self.pipeline.predict(query)[0])

        proba = self.pipeline.predict_proba(query)[0]
        classes = list(self.pipeline.named_steps["knn"].classes_)
        label_idx = classes.index(predicted_label)
        confidence_score = float(proba[label_idx])

        knn: KNeighborsClassifier = self.pipeline.named_steps["knn"]
        query_scaled = self._transform_query(query)
        distances, _ = knn.kneighbors(query_scaled, n_neighbors=knn.n_neighbors, return_distance=True)
        nearest_distances = [round(float(d), 4) for d in distances[0]]

        return predicted_label, round(confidence_score, 4), nearest_distances

    def explain_neighbors(
        self,
        neck_angle: float,
        torso_angle: float,
        shoulder_tilt: float,
    ) -> list[dict[str, Any]]:
        """
        Return detailed information about the K nearest neighbors in feature space
        for the Explainable AI (XAI) panel in the CSC472 demo.
        """
        query = np.array([[float(neck_angle), float(torso_angle), float(shoulder_tilt)]], dtype=float)
        knn: KNeighborsClassifier = self.pipeline.named_steps["knn"]

        query_scaled = self._transform_query(query)
        distances, indices = knn.kneighbors(query_scaled, n_neighbors=knn.n_neighbors, return_distance=True)

        neighbors_info: list[dict[str, Any]] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            row = self.dataset_df.iloc[int(idx)]
            neighbors_info.append(
                {
                    "rank": rank,
                    "euclidean_distance": round(float(dist), 4),
                    "label": str(row["label"]),
                    "neck_angle": round(float(row["neck_angle"]), 1),
                    "torso_angle": round(float(row["torso_angle"]), 1),
                    "shoulder_tilt": round(float(row["shoulder_tilt"]), 4),
                }
            )
        return neighbors_info

    def add_calibration_sample(
        self,
        neck_angle: float,
        torso_angle: float,
        shoulder_tilt: float,
        label: str,
        replicas: int = 1,
    ) -> int:
        """
        Append live user calibration sample(s) to the CSV dataset and retrain immediately.
        Returns the updated dataset row count.
        """
        if label not in POSTURE_CLASSES:
            raise ValueError(f"Invalid posture label '{label}'. Must be one of {POSTURE_CLASSES}")

        ensure_dataset_exists(self.csv_path)
        df = pd.read_csv(self.csv_path)

        # Remove conflicting rows within the local neighborhood of this calibrated posture
        # so old/synthetic samples at the same angle never out-vote user calibration
        if replicas > 1:
            conflict_mask = (
                (df["label"] != label)
                & ((df["neck_angle"] - float(neck_angle)).abs() <= 3.5)
                & ((df["torso_angle"] - float(torso_angle)).abs() <= 4.5)
            )
            df = df[~conflict_mask].reset_index(drop=True)

        new_rows: list[dict[str, float | str]] = [
            {
                "neck_angle": round(float(neck_angle), 2),
                "torso_angle": round(float(torso_angle), 2),
                "shoulder_tilt": round(float(shoulder_tilt), 4),
                "label": label,
            }
        ]

        if replicas > 1:
            rng = np.random.default_rng(101)
            for _ in range(replicas - 1):
                new_rows.append(
                    {
                        "neck_angle": round(float(neck_angle + rng.normal(0.0, 0.35)), 2),
                        "torso_angle": round(float(torso_angle + rng.normal(0.0, 0.45)), 2),
                        "shoulder_tilt": round(max(0.0, float(shoulder_tilt + rng.normal(0.0, 0.002))), 4),
                        "label": label,
                    }
                )

        updated_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        updated_df.to_csv(self.csv_path, index=False)
        self.train(self.csv_path)
        return len(self.dataset_df)

    def reset_to_seed_dataset(self) -> int:
        """Reset dataset back to the initial 300 synthetic samples and retrain."""
        ensure_dataset_exists(self.csv_path, force_regenerate=True)
        self.train(self.csv_path)
        return len(self.dataset_df)
