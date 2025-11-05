from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor


class SparseScalerTorch(BaseEstimator, TransformerMixin):
    """SparseScaler PyTorch implementation."""

    def __init__(self, mask: bool = True, exponent: int = 4):
        self.mask = mask
        self.exponent = exponent
        self._fitted = False

    def fit(self, X: TorchTensor) -> Self:
        if self._fitted:
            raise RuntimeError("SparseScaler is already fitted.")

        # Square-root transform the features
        X = X.clamp(min=0).sqrt()

        # Per-feature sparsity and epsilon
        alpha = (X == 0).float().mean(dim=0)
        self.epsilon = alpha**self.exponent + 1e-8

        # Per-feature mean and std
        self.mu = X.mean(dim=0)
        self.sigma = X.std(dim=0) + self.epsilon

        self._fitted = True
        return self

    def transform(self, X: TorchTensor) -> TorchTensor:
        if not self._fitted:
            raise RuntimeError("SparseScaler is not fitted yet. Call 'fit()' method first.")

        # Square-root transform the features
        X = X.clamp(min=0).sqrt()

        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        return (X - self.mu) / self.sigma

    def fit_transform(self, X: TorchTensor) -> TorchTensor:
        self.fit(X)
        return self.transform(X)


class SparseScalerNumpy(BaseEstimator, TransformerMixin):
    """SparseScaler NumPy implementation."""

    def __init__(self, mask: bool = True, exponent: int = 4):
        self.mask = mask
        self.exponent = exponent
        self._fitted = False

    def fit(self, X: np.ndarray) -> Self:
        if self._fitted:
            raise RuntimeError("SparseScalerNumpy is already fitted.")

        # Square-root transform the features
        X = np.sqrt(np.clip(X, a_min=0, a_max=None))

        # Per-feature sparsity and epsilon
        alpha = (X == 0).astype(float).mean(axis=0)
        self.epsilon = alpha**self.exponent + 1e-8

        # Per-feature mean and std
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + self.epsilon

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SparseScalerNumpy is not fitted yet. Call 'fit()' method first.")

        # Square-root transform the features
        X = np.sqrt(np.clip(X, a_min=0, a_max=None))

        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        return (X - self.mu) / self.sigma

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
