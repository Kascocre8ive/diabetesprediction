"""
Hybrid soft-voting ensemble that combines the three base models
(SVM, Logistic Regression, XGBoost) using weighted-average probabilities.

This class is intentionally simple and dependency-free (pure numpy) so it
can be pickled with joblib and reloaded both by the training script and by
the Streamlit app without needing scikit-learn's VotingClassifier (which
would require re-fitting on a single combined training call).
"""
import numpy as np


class SoftVotingEnsemble:
    """Weighted soft-voting ensemble over a list of fitted classifiers.

    Parameters
    ----------
    models : list
        Fitted classifiers, each exposing `predict_proba(X)`.
    weights : list[float], optional
        One weight per model (same order as `models`). Defaults to equal
        weights. Weights are normalized internally, so (1, 1, 2) and
        (0.25, 0.25, 0.5) behave identically.
    names : list[str], optional
        Human-readable names for each model, in the same order as
        `models`. Defaults to "model_0", "model_1", ...
    """

    def __init__(self, models, weights=None, names=None):
        if not models:
            raise ValueError("SoftVotingEnsemble requires at least one model.")

        self.models = list(models)
        self.weights = list(weights) if weights is not None else [1.0] * len(models)
        self.names = list(names) if names is not None else [f"model_{i}" for i in range(len(models))]

        if len(self.weights) != len(self.models):
            raise ValueError("weights must have the same length as models.")
        if len(self.names) != len(self.models):
            raise ValueError("names must have the same length as models.")

    def per_model_proba(self, X):
        """Return {model_name: array of P(class=1)} for every base model."""
        return {
            name: np.asarray(model.predict_proba(X))[:, 1]
            for name, model in zip(self.names, self.models)
        }

    def predict_proba(self, X):
        """Return an (n_samples, 2) array of [P(class=0), P(class=1)]."""
        probas = np.array([
            np.asarray(model.predict_proba(X))[:, 1] for model in self.models
        ])  # shape: (n_models, n_samples)

        weights = np.asarray(self.weights, dtype=float).reshape(-1, 1)
        weighted_sum = (probas * weights).sum(axis=0)
        weighted_avg = weighted_sum / weights.sum()

        return np.column_stack([1 - weighted_avg, weighted_avg])

    def predict(self, X, threshold=0.5):
        """Return hard class predictions (0/1) using the given threshold."""
        proba_pos = self.predict_proba(X)[:, 1]
        return (proba_pos >= threshold).astype(int)

    def get_params(self):
        """Convenience accessor mirroring sklearn's get_params-style API."""
        return {"weights": self.weights, "names": self.names}