"""Channel-Aware Sampling — multi-scale view generation for DINO.

Global views: ceil(70%) channels, 80% temporal window.
Local views:  floor(30%) channels, 50% temporal window.
Masked views: global crop + 20% channel & temporal patch masking.

ceil() for global channels preserves asymmetry with small channel counts:
  C=2 → global=2ch (100%), local=1ch (50%).
  C=19 → global=14ch (74%), local=5ch (26%).
"""
import math
import torch
import numpy as np


class ChannelAwareSampling:

    def __init__(self, n_channels=2, sampling_rate=200,
                 n_local_views=4, n_masked_views=1):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.n_local_views = n_local_views
        self.n_masked_views = n_masked_views

    def _random_crop(self, x, ch_frac, time_frac, ceil_ch=False):
        B, C, T = x.shape
        n_t = int(time_frac * T)
        t_start = np.random.randint(0, T - n_t + 1)

        C_eff = min(C, self.n_channels)
        n_ch = max(1, math.ceil(ch_frac * C_eff) if ceil_ch else int(ch_frac * C_eff))
        n_ch = min(n_ch, C_eff)
        ch_idx = np.sort(np.random.choice(C_eff, n_ch, replace=False))

        return x[:, ch_idx, t_start:t_start + n_t], torch.LongTensor(ch_idx)

    def create_global_view(self, x):
        return self._random_crop(x, 0.7, 0.8, ceil_ch=True)

    def create_local_view(self, x):
        return self._random_crop(x, 0.3, 0.5, ceil_ch=False)

    def create_masked_view(self, x):
        view, ch_idx = self.create_global_view(x)
        _, C, T = view.shape

        if C > 1:
            n_mask = max(1, int(0.2 * C))
            view[:, np.random.choice(C, n_mask, replace=False), :] = 0

        ps = self.sampling_rate
        n_patches = T // ps
        n_mask_t = max(1, int(0.2 * n_patches))
        for p in np.random.choice(n_patches, n_mask_t, replace=False):
            view[:, :, p * ps:(p + 1) * ps] = 0

        return view, ch_idx

    def __call__(self, x):
        views = {}
        for i in range(2):
            v, ch = self.create_global_view(x)
            views[f'global_{i}'] = {'view': v, 'channels': ch}
        for i in range(self.n_local_views):
            v, ch = self.create_local_view(x)
            views[f'local_{i}'] = {'view': v, 'channels': ch}
        for i in range(self.n_masked_views):
            v, ch = self.create_masked_view(x)
            views[f'masked_{i}'] = {'view': v, 'channels': ch}
        return views
