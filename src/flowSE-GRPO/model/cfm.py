from __future__ import annotations
import sys
import os
sys.path.append(os.path.dirname(__file__))
from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint

from model.backbones.sde_sampler import FlowSESampler
from modules import MelSpec
from model.model_utils import default, exists


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(method="euler"),
        audio_drop_prob=0.0,  
        cond_drop_prob=0.0,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
    ):
        super().__init__()

        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.dim = transformer.dim
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs

    @property
    def device(self):
        return next(self.parameters()).device

    def sample_rl(
        self,
        cond: torch.Tensor,
        *,
        steps=10, 
        cfg_strength=0.0,
        sde_kwargs: dict = dict(noise_level_a=0.4, sde_window_start=1, sde_window_size=2)
    ):
        """SDE sampling optimized for GRPO data collection."""
        self.eval()

        if cond.ndim == 2:
            cond = self.mel_spec(cond).permute(0, 2, 1)

        cond = cond.to(next(self.parameters()).dtype)
        step_cond = cond
        mask = None

        def fn(t, x):
            pred = self.transformer(
                x=x, cond=step_cond, time=t, mask=mask, drop_audio_cond=False
            )
            if cfg_strength < 1e-5:
                return pred
            null_pred = self.transformer(
                x=x, cond=step_cond, time=t, mask=mask, drop_audio_cond=True
            )
            return pred + (pred - null_pred) * cfg_strength

        y0 = torch.randn_like(cond)
        sampler = FlowSESampler(steps=steps, **sde_kwargs)
        out_mel, trajectory, rl_states = sampler.sample(model_fn=fn, y0=y0)

        return out_mel, trajectory, rl_states
    
    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        *,
        steps=12,
        cfg_strength=0.0,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
    ):
        self.eval()

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
     
        step_cond = cond
        mask = None

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        def fn(t, x):
            pred = self.transformer(
                x=x, cond=step_cond, time=t, mask=mask, drop_audio_cond=False
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, time=t, mask=mask, drop_audio_cond=True
            )
            return pred + (pred - null_pred) * cfg_strength

        y0 = torch.randn_like(cond)
        t_start = 0
        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        sampled = trajectory[-1]
        out = sampled

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward_rl(
        self,
        x_t: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        x_next: torch.Tensor,
        dt: float,
        a: float = 0.4,
        cfg_strength: float = 0.0
    ):
        """
        Forward method for GRPO training. Computes the log probability 
        of the action (x_next) based on the SDE sampling trajectory.
        """
        t_b = t.view(-1, 1, 1)

        pred = self.transformer(
            x=x_t, cond=cond, time=t, mask=None, drop_audio_cond=False
        )
        
        if cfg_strength > 1e-5:
            null_pred = self.transformer(
                x=x_t, cond=cond, time=t, mask=None, drop_audio_cond=True
            )
            v_theta = pred + (pred - null_pred) * cfg_strength
        else:
            v_theta = pred

        # Recompute SDE mean (Equation 6)
        sigma_t = a * torch.sqrt((1.0 - t_b) / t_b)
        drift_correction = (sigma_t ** 2) / (2 * (1.0 - t_b)) * (-x_t + t_b * v_theta)
        x_mean = x_t + (v_theta + drift_correction) * dt
        std_dev = sigma_t * torch.sqrt(torch.tensor(dt, device=x_t.device))
        
        with torch.cuda.amp.autocast(enabled=False):
            dist = torch.distributions.Normal(x_mean.float(), std_dev.float())
            log_prob = dist.log_prob(x_next.float())
            log_prob = log_prob.mean(dim=(1, 2))
        
        return log_prob, x_mean, std_dev
    
    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # noqa: F722 
        clean: float["b n d"] | float["b nw"],
    ):
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            
            clean = self.mel_spec(clean)
            clean = clean.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, _, dtype, device, _ = *inp.shape[:2], inp.dtype, self.device, self.sigma
    
        x1 = clean
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=self.device)

        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        cond = inp
        
        drop_audio_cond = random() < self.audio_drop_prob 
        if random() < self.cond_drop_prob: 
            drop_audio_cond = True

        pred = self.transformer(
            x=φ, cond=cond, time=time, drop_audio_cond=drop_audio_cond,
        )

        loss = F.mse_loss(pred, flow, reduction="none")

        return loss.mean(), cond, pred