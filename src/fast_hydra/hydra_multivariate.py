"""
HYDRA multivariate feature extractor.

- Original code by Angus Dempster
- Extended to PyTorch CPU/GPU support by Nima Nooshiri

Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). Hydra: Competing
convolutional kernels for fast and accurate time series classification. Data
Mining and Knowledge Discovery, 37(5), 1779-1805.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HydraMultivariate(nn.Module):
    dilations: torch.Tensor
    paddings: torch.Tensor

    kernel_size: int = 9

    def __init__(
        self,
        input_size: int,
        n_channels: int,
        *,
        n_groups: int = 64,
        n_kernels: int = 8,
        max_num_channels: int = 8,
        max_num_dilations: int | None = None,
        random_state: int | None = None,
    ):
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)

        super().__init__()

        self.input_size: int = input_size
        self.n_channels: int = n_channels
        self.n_groups: int = n_groups
        self.n_kernels: int = n_kernels
        self.max_num_channels: int = max_num_channels
        self.rand_seed: int | None = random_state

        self.max_exponent: int = self._get_max_exponent()

        # If g > 1, assign half the groups to X and half the groups to diff(X)
        self._divisor: int = 2 if self.n_groups > 1 else 1
        self._h: int = self.n_groups // self._divisor

        self.register_dilations(max_num_dilations)
        self.register_paddings()
        self.register_dilation_weights()
        self.register_channel_selectors()

    def _get_max_exponent(self) -> int:
        return int(np.log2((self.input_size - 1) / (self.kernel_size - 1)))

    def register_dilations(self, max_num_dilations: int | None = None) -> None:
        dilations = torch.pow(2, torch.arange(self.max_exponent + 1))
        self.register_buffer("dilations", dilations[:max_num_dilations])

    def register_paddings(self) -> None:
        paddings = torch.div(
            (self.kernel_size - 1) * self.dilations, 2, rounding_mode="floor"
        ).int()
        self.register_buffer("paddings", paddings)

    @property
    def n_dilations(self) -> int:
        return len(self.dilations)

    def register_dilation_weights(self) -> None:
        for i_dilation in range(self.n_dilations):
            w_ = torch.randn(self._divisor, self.k * self._h, 1, self.kernel_size)
            w_ -= w_.mean(-1, keepdim=True)
            w_ /= w_.abs().sum(-1, keepdim=True)
            # Register each as a buffer so they move with the model
            self.register_buffer(f"dilation_weight_{i_dilation}", w_)

    def register_channel_selectors(self) -> None:
        # Combine num_channels // 2 channels (2 < n < max_num_channels)
        num_channels_per = np.clip(self.n_channels // 2, 2, self.max_num_channels)

        for i_dilation in range(self.n_dilations):
            i_ = torch.randint(
                low=0,
                high=self.n_channels,
                size=(self._divisor, self._h, num_channels_per),
            )
            self.register_buffer(f"channel_selector_{i_dilation}", i_)

    @property
    def g(self) -> int:
        return self.n_groups

    @property
    def k(self) -> int:
        return self.n_kernels

    @property
    def W(self) -> list[torch.Tensor]:
        # Should return registered buffers instead of the list
        return [getattr(self, f"dilation_weight_{i}") for i in range(self.n_dilations)]

    @property
    def I(self) -> list[torch.Tensor]:  # noqa: E743
        return [getattr(self, f"channel_selector_{i}") for i in range(self.n_dilations)]

    def batch(self, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self.forward(X)

        batches = torch.arange(num_examples).split(batch_size)
        Z = [self.forward(X[batch]) for batch in batches]
        return torch.cat(Z)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        num_examples = X.shape[0]

        diff_X = torch.diff(X, n=1, dim=-1) if self.n_groups > 1 else X

        Z = []
        for dilation_index in range(self.n_dilations):
            d = int(self.dilations[dilation_index])
            p = int(self.paddings[dilation_index])

            for diff_index in range(min(2, self.n_groups)):
                # diff_index == 0 -> use X
                # diff_index == 1 -> use diff(X)
                dummy_X = X if diff_index == 0 else diff_X
                dummy_Z = F.conv1d(
                    input=dummy_X[:, self.I[dilation_index][diff_index]].sum(2),
                    weight=self.W[dilation_index][diff_index],
                    dilation=d,
                    padding=p,
                    groups=self._h,
                ).view(num_examples, self._h, self.k, -1)

                max_values, max_indices = dummy_Z.max(2)
                count_max = torch.zeros(num_examples, self._h, self.k, device=X.device)

                min_values, min_indices = dummy_Z.min(2)
                count_min = torch.zeros(num_examples, self._h, self.k, device=X.device)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z
