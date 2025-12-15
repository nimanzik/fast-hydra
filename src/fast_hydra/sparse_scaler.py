from __future__ import annotations

from typing import TYPE_CHECKING, Self, TypeGuard, overload

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor as TorchTensor


class SparseScaler(BaseEstimator, TransformerMixin):
    """Sparse-aware feature scaler for PyTorch tensors and NumPy arrays.

    This scaler applies a square-root transformation followed by standardisation, with
    adjustments based on feature sparsity. It is designed for feature spaces where many
    values can be zero. The scaler automatically detects whether the input is a PyTorch
    tensor or NumPy array and uses the appropriate backend operations.

    The scaling process includes:
    1. Square-root transformation to reduce skewness
    2. Sparsity-aware epsilon calculation for numerical stability
    3. Standardisation with optional masking of zero values

    Parameters
    ----------
    mask : bool, default=True
        If True, preserve zero values during transformation by applying a mask that
        sets (x - mean) to zero where x is zero.
    exponent : int, default=4
        Exponent applied to the sparsity ratio when calculating epsilon. Higher values
        reduce the epsilon for denser features.

    Attributes
    ----------
    epsilon_ : torch.Tensor or numpy.ndarray
        Sparsity-adjusted stabilisation term for each feature.
    mu_ : torch.Tensor or numpy.ndarray
        Per-feature mean of square-root transformed data.
    sigma_ : torch.Tensor or numpy.ndarray
        Per-feature standard deviation with epsilon adjustment.

    Examples
    --------
    >>> import numpy as np
    >>> scaler = SparseScaler(mask=True, exponent=4)
    >>> X_train = np.array([[0., 1., 4.], [0., 0., 9.], [1., 2., 16.]])
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)
    >>> X_test = np.array([[0., 1., 4.]])
    >>> X_test_scaled = scaler.transform(X_test)
    >>>
    >>> import torch
    >>> scaler = SparseScaler(mask=True, exponent=4)
    >>> X_train = torch.tensor([[0., 1., 4.], [0., 0., 9.], [1., 2., 16.]])
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)
    """

    def __init__(self, mask: bool = True, exponent: int = 4) -> None:
        """Initialise the sparse scaler with specified parameters."""
        self.mask = mask
        self.exponent = exponent
        self._is_fitted = False
        self._backend: str | None = None

    def _is_torch_tensor(self, X: TorchTensor | NDArray) -> TypeGuard[TorchTensor]:
        """Check if input is a PyTorch tensor."""
        return hasattr(X, "__torch_function__")

    def fit(self, X: TorchTensor | NDArray) -> Self:
        """Compute scaling parameters from training data.

        Parameters
        ----------
        X : torch.Tensor or numpy.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        self : object
            Fitted scaler instance.
        """
        if self._is_fitted:
            raise RuntimeError("SparseScaler is already fitted")

        # Detect backend and compute scaling parameters
        if self._is_torch_tensor(X):
            # PyTorch backend
            self._backend = "torch"
            X_hat = X.clamp(min=0).sqrt()
            alpha = (X_hat == 0).float().mean(dim=0)
            self.epsilon_ = alpha**self.exponent + 1e-8
            self.mu_ = X_hat.mean(dim=0)
            self.sigma_ = X_hat.std(dim=0) + self.epsilon_
        else:
            # NumPy backend
            self._backend = "numpy"
            X_hat = np.sqrt(np.clip(X, a_min=0, a_max=None))
            alpha = (X_hat == 0).astype(float).mean(axis=0)
            self.epsilon_ = alpha**self.exponent + 1e-8
            self.mu_ = X_hat.mean(axis=0)
            self.sigma_ = X_hat.std(axis=0) + self.epsilon_

        self._is_fitted = True
        return self

    @overload
    def transform(self, X: TorchTensor) -> TorchTensor: ...

    @overload
    def transform(self, X: NDArray) -> NDArray: ...

    def transform(self, X: TorchTensor | NDArray) -> TorchTensor | NDArray:
        """Apply learnt scaling transformation to data.

        Parameters
        ----------
        X : torch.Tensor or numpy.ndarray
            Data to transform of shape (n_samples, n_features).

        Returns
        -------
        X_hat : torch.Tensor or numpy.ndarray
            Scaled data with same shape as input.
        """
        if not self._is_fitted:
            raise RuntimeError("SparseScaler is not fitted. Call 'fit()' first")

        # Check backend consistency
        is_torch = self._is_torch_tensor(X)
        this_backend = "torch" if is_torch else "numpy"
        if this_backend != self._backend:
            raise TypeError(
                f"Input type mismatch: scaler was fitted with {self._backend} "
                f"but received {this_backend} data"
            )

        if self._is_torch_tensor(X):
            # PyTorch backend
            X_hat = X.clamp(min=0).sqrt()
        else:
            # NumPy backend
            X_hat = np.sqrt(np.clip(X, a_min=0, a_max=None))

        if self.mask:
            return ((X_hat - self.mu_) * (X_hat != 0)) / self.sigma_
        return (X_hat - self.mu_) / self.sigma_

    @overload
    def fit_transform(
        self, X: TorchTensor, y: None = None, **fit_params: object
    ) -> TorchTensor: ...

    @overload
    def fit_transform(
        self, X: NDArray, y: None = None, **fit_params: object
    ) -> NDArray: ...

    def fit_transform(
        self, X: TorchTensor | NDArray, y: None = None, **fit_params: object
    ) -> TorchTensor | NDArray:
        """Fit the scaler and transform data in one step.

        Parameters
        ----------
        X : torch.Tensor or numpy.ndarray
            Training data of shape (n_samples, n_features).
        y : None, default=None
            Ignored. Present for scikit-learn API compatibility.
        **fit_params : object
            Additional fit parameters (ignored).

        Returns
        -------
        X_hat : torch.Tensor or numpy.ndarray
            Scaled training data.
        """
        self.fit(X)
        return self.transform(X)
