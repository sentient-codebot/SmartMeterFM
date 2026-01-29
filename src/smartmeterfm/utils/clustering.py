import enum
import pickle

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class AllowedDataTypes(str, enum.Enum):
    NUMPY = "numpy"
    TENSOR = "tensor"


class ClusteringModel:
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.is_fitted = False

    @staticmethod
    def to_numpy(data):
        if isinstance(data, np.ndarray):
            return data, AllowedDataTypes.NUMPY
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy(), AllowedDataTypes.TENSOR
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. Expected numpy.ndarray or torch.Tensor."
            )

    def fit(self, X: np.ndarray):
        """Fit the clustering model to data.

        Args:
            X (array-like): Input data to fit the model. [n_samples, n_features]

        """
        if self.n_clusters is None:
            self.n_clusters = self._find_optimal_clusters(X)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict cluster labels for new data points.

        Args:
            X (array-like): New data points to predict cluster labels for. [n_samples, n_features]

        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        array_X, in_type = self.to_numpy(X)
        out = self.kmeans.predict(array_X)
        if in_type == AllowedDataTypes.TENSOR:
            out = torch.tensor(out, dtype=X.dtype, device=X.device)
        return out

    def get_cluster_info(self, point):
        """Get cluster assignment and center for a single point."""
        point, _in_type = self.to_numpy(point)
        point = np.array(point).reshape(1, -1)
        cluster_id = self.kmeans.predict(point)[0]
        cluster_center = self.kmeans.cluster_centers_[cluster_id]
        return cluster_id, cluster_center

    def _find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using silhouette score."""
        scores = []
        K_range = range(2, min(max_clusters + 1, len(X)))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append(score)

        return K_range[np.argmax(scores)]

    def save_model(self, filepath):
        """Save the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        """Load a fitted model."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
