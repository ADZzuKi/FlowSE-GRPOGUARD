import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as tf
import torchaudio




class STFT(nn.Module):
    def __init__(self, win_len, hop_len, fft_len, win_type):
        super(STFT, self).__init__()
        self.win, self.hop = win_len, hop_len
        self.nfft = fft_len
        self.win_type = win_type

        
        window = {
            "hann": th.hann_window(win_len),
            "hamm": th.hamming_window(win_len),
        }
        assert self.win_type in window.keys()
        self.window_analysis = window[self.win_type]
        self.window_synthesis = window[self.win_type]

    def transform(self, inp):
        """
        inp: N x L
        out: N x F x T x C
        """
      
        cspec = th.stft(
            inp,
            self.nfft,
            self.hop,
            self.win,
            self.window_analysis.to(inp.device),
            return_complex=True,
        )
        real = cspec.real
        imag = cspec.imag
        cspec = th.stack([real, imag], dim=-1)
        # cspec = th.einsum('nftc->ncft', cspec)
        return cspec

    def inverse(self, real, imag):
        """
        real, imag: N x F x T
        return: N x L
        """

        inp = th.stack([real, imag], dim=-1)
        inp = torch.view_as_complex(inp)
        return th.istft(
            inp, self.nfft, self.hop, self.win, self.window_synthesis.to(real.device)
        )


