"""Channel-Aware Sampling — creates multi-scale views for DINO."""
import torch
import numpy as np


class ChannelAwareSampling:
    def __init__(self, n_channels=2, sampling_rate=200,
                 n_local_views=4, n_masked_views=1):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.n_local_views = n_local_views
        self.n_masked_views = n_masked_views

    def _random_crop(self, x, ch_frac, time_frac):
        """Select ch_frac channels and time_frac temporal window."""
        B, C, T = x.shape
        n_ch = max(1, int(ch_frac * min(C, self.n_channels)))
        n_t = int(time_frac * T)
        ch_idx = np.sort(np.random.choice(min(C, self.n_channels), n_ch, replace=False))
        t_start = np.random.randint(0, T - n_t + 1)
        return x[:, ch_idx, t_start:t_start + n_t], torch.LongTensor(ch_idx)

    def create_global_view(self, x):
        return self._random_crop(x, 0.7, 0.8)

    def create_local_view(self, x):
        return self._random_crop(x, 0.3, 0.5)

    def create_masked_view(self, x):
        view, ch_idx = self.create_global_view(x)
        _, C, T = view.shape

        # Channel masking (20%), skip if only 1 channel
        if C > 1:
            n_mask = max(1, int(0.2 * C))
            view[:, np.random.choice(C, n_mask, replace=False), :] = 0

        # Temporal patch masking (20% of 1-second patches)
        ps = self.sampling_rate
        n_patches = T // ps
        for p in np.random.choice(n_patches, max(1, int(0.2 * n_patches)), replace=False):
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
