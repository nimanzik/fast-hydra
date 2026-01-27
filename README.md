# fast-hydra

***GPU-optimised PyTorch implementation of HYDRA (HYbrid Dictionary-Rocket Architecture) for fast and accurate time-series classification.***

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **GPU-accelerated** feature extraction for multivariate time series.
- **SparseScaler**: sparsity-aware feature scaling for both PyTorch and NumPy.
- Scikit-learn compatible API.
- Competitive convolutional kernels with configurable number of dilations to control number of features to extract.

## Installation

```bash
uv add git+https://github.com/nimanzik/fast-hydra.git
```

## Quick Start

```python
import torch
from fast_hydra import HydraMultivariate, SparseScaler

input_size = 1000  # Length of time series
n_channels = 6   # Number of channels in multivariate time series
batch_size = 32  # Number of samples in a batch

# Extract features from multivariate time series
hydra = HydraMultivariate(
    input_size=input_size,
    n_channels=n_channels,
    n_groups=64,
    n_kernels=8,
    random_state=42,
)
X = torch.randn(batch_size, n_channels, input_size)
features = hydra(X)  # Tensor of shape (batch_size, n_features)

# Scale sparse features
scaler = SparseScaler()
scaled_features = scaler.fit_transform(features)
```

## Requirements

- Numpy >= 2.3.4
- scikit-learn >= 1.7.2
- PyTorch >= 2.9.0

## Citation

Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). Hydra: Competing convolutional kernels for fast and accurate time series classification. Data Mining and Knowledge Discovery, 37(5), 1779-1805.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
