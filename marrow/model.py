#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marrow Behavioral Biometrics: Classifier & Verification Engine
Evaluates behavioral biometric match using robust regularized Mahalanobis distance.
"""
import math
from typing import Dict, List, Tuple
from .features import FeatureExtractor
from .format import SessionRecording


class BehavioralProfile:
    """Statistical behavioral profile of an enrolled user with fixed decision threshold."""

    def __init__(self, user_label: str, threshold_multiplier: float = 3.0):
        self.user_label = user_label
        self.k_multiplier = threshold_multiplier
        self.means: List[float] = []
        self.stds: List[float] = []
        self.threshold: float = 0.0
        self.is_enrolled: bool = False

    def train(self, training_sessions: List[SessionRecording]) -> None:
        if not training_sessions:
            raise ValueError("Training requires at least one enrollment session.")

        feature_matrix = [FeatureExtractor.extract_features(s) for s in training_sessions]
        n_features = len(FeatureExtractor.FEATURE_NAMES)
        n_samples = len(feature_matrix)

        self.means = []
        self.stds = []

        for j in range(n_features):
            col = [feature_matrix[i][j] for i in range(n_samples)]
            col_mean = sum(col) / max(1, n_samples)
            if n_samples > 1:
                var = sum((x - col_mean) ** 2 for x in col) / (n_samples - 1)
                std = math.sqrt(var)
            else:
                std = abs(col_mean) * 0.15 + 1e-4

            std_reg = max(std, abs(col_mean) * 0.15 + 1e-4)
            self.means.append(col_mean)
            self.stds.append(std_reg)

        # Identify active features with signal
        self.active_indices = [
            j for j in range(n_features) if self.means[j] > 1e-3 or self.stds[j] > 1e-4
        ]

        # Fix decision threshold from training data before testing
        train_distances = [self._compute_distance(f) for f in feature_matrix]
        max_train_dist = max(train_distances)
        self.threshold = round(max(1.65, max_train_dist * 1.35), 4)
        self.is_enrolled = True

    def _compute_distance(self, features: List[float]) -> float:
        indices = getattr(self, "active_indices", list(range(len(features))))
        z_scores = [
            abs(features[j] - self.means[j]) / self.stds[j]
            for j in indices
        ]
        if not z_scores:
            return 0.0

        # Quadratic mean of top-3 discriminative deviations + average z-score
        z_sorted = sorted(z_scores, reverse=True)
        top3_rms = math.sqrt(sum(z ** 2 for z in z_sorted[:3]) / min(3, len(z_sorted)))
        avg_z = sum(z_scores) / len(z_scores)
        return round(0.50 * avg_z + 0.50 * top3_rms, 4)

    def evaluate_session(self, session: SessionRecording) -> Dict[str, Any]:
        if not self.is_enrolled:
            raise RuntimeError("Profile must be enrolled before evaluation.")
        features = FeatureExtractor.extract_features(session)
        score = round(self._compute_distance(features), 4)
        accepted = score <= self.threshold
        return {
            "session_id": session.session_id,
            "subject_label": session.subject_label,
            "score": score,
            "threshold": self.threshold,
            "accepted": accepted,
            "decision": "ACCEPT_OWNER" if accepted else "REJECT_IMPOSTOR",
        }
