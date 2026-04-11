

"""
删除了文本condition的逻辑，只用audio作condition
"""
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

from modules import MelSpec
from model.model_utils import (
    default,
    exists,
    list_str_to_idx,
    list_str_to_tensor,
)


class CFM(nn.Module):
   
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),


        audio_drop_prob=0.0,  
        cond_drop_prob=0.0,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        # vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # # vocab map for tokenization
        # self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device


    '''
        cond: noisy speech
        text: transcription
    '''

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        # text: int["b nt"] | list[str],  # noqa: F722
      
        *,
        steps=32,
        cfg_strength=1.0,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        # drop_text=False
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # # text

        # if isinstance(text, list):
        #     if exists(self.vocab_char_map):
        #         text = list_str_to_idx(text, self.vocab_char_map).to(device)
        #     else:
        #         text = list_str_to_tensor(text).to(device)
        #     assert text.shape[0] == batch

        # if exists(text):
        #     text_lens = (text != -1).sum(dim=-1)
        #     lens = torch.maximum(text_lens, lens)  # make sure lengths are at least those of the text characters
     
        step_cond = cond
        
        mask = None

        if no_ref_audio:
            cond = torch.zeros_like(cond)


        def fn(t, x):
    
            pred = self.transformer(
                x=x, cond=step_cond,  time=t, mask=mask, drop_audio_cond=False
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, time=t, mask=mask, drop_audio_cond=True
            )
            return pred + (pred - null_pred) * cfg_strength

        y0 = torch.randn_like(cond)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722 
        clean: float["b n d"] | float["b nw"],
        # text: int["b nt"] | list[str],  # noqa: F722
    ):
        '''
        inp: noisy speech
        clean: clean speech
        text: transcription
        
        '''
        
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            
            clean = self.mel_spec(clean)
            clean = clean.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, _ , dtype, device, _ = *inp.shape[:2], inp.dtype, self.device, self.sigma
    
        # if isinstance(text, list):
        #     if exists(self.vocab_char_map):
        #         text = list_str_to_idx(text, self.vocab_char_map).to(device)
        #     else:
        #         text = list_str_to_tensor(text).to(device)
        #     assert text.shape[0] == batch

        # mel is x1
        x1 = clean

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        # cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        cond = inp
        
        
        # transformer and cfg training with a drop rate

        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True


        pred = self.transformer(
            x=φ, cond=cond, time=time, drop_audio_cond=drop_audio_cond,
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")

        return loss.mean(), cond, pred


if __name__ == "__main__":
    model = CFM()
    