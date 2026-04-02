import os
import sys
import argparse
import random
import re
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.tensorboard import SummaryWriter
import onnxruntime as ort

from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from vocos import Vocos

from utils.logger import get_logger
from model import DiT, CFM
from loader.dataloader import make_auto_loader
from GRPOTrainer import GRPOTrainer
from tqdm import tqdm
from utils.ema import EMAModuleWrapper


class GPUDNSMOSFeature(torch.nn.Module):
    def __init__(self, sr=16000, n_mels=120, frame_size=320, hop_length=160):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=frame_size+1, win_length=frame_size,
            hop_length=hop_length, n_mels=n_mels, center=True, pad_mode="reflect"
        )
        self.resampler = torchaudio.transforms.Resample(24000, sr)


class OmniRewardWrapper(nn.Module):
    """
    Reward model wrapper integrating DNSMOS (and extensible to SPK/WER).
    Provides scalar reward signals for GRPO policy optimization.
    """
    def __init__(self, dnsmos_dir, weights):
        super().__init__()
        providers = ['CUDAExecutionProvider']
        self.dns_sess = ort.InferenceSession(f"{dnsmos_dir}/sig_bak_ovr.onnx", providers=providers)
        self.feature_extractor = GPUDNSMOSFeature().cuda()
        self.p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
        self.p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
        self.p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
        
        self.gt_emb_cache = {}
        self.weights = weights

    def get_spk_embedding(self, wav_tensor_1d):
        """Extract speaker embedding directly from tensor (avoiding IO overhead)."""
        wav = wav_tensor_1d.unsqueeze(0).cpu() 
        wav = wav * (1 << 15)                  
        feats = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10, energy_floor=0.0, sample_frequency=16000)
        feats = feats - torch.mean(feats, dim=0) 
        feats = feats.unsqueeze(0).cuda()        
        
        with torch.no_grad():
            outputs = self.spk_model(feats)
            emb = outputs[-1] if isinstance(outputs, tuple) else outputs
        return emb.squeeze()

    def forward(self, audio_24k_tensor, clean_24k_tensor):
        with torch.cuda.amp.autocast(enabled=False):
            audio_16k = self.feature_extractor.resampler(audio_24k_tensor).to(torch.float32)
            clean_16k = self.feature_extractor.resampler(clean_24k_tensor).to(torch.float32)
            B = audio_16k.size(0)
            len_samples = int(9.01 * 16000)
            
            w_ovr = self.weights.get("ovrl", 1.0)

            total_rewards = []
            log_ovrl = []
            
            for i in range(B):
                aud = audio_16k[i].squeeze().cpu().numpy()
                c_aud_tensor = clean_16k[i].squeeze()

                if len(aud) < len_samples:
                    aud = np.pad(aud, (0, len_samples - len(aud)), 'wrap')
                
                num_hops = max(1, int(np.floor(len(aud) / 16000) - 9.01) + 1)
                ovr_l = []
                for hop in range(num_hops):
                    seg = aud[hop*16000 : hop*16000+len_samples]
                    if len(seg) < len_samples: 
                        seg = np.pad(seg, (0, len_samples-len(seg)), 'wrap')
                    oi = {self.dns_sess.get_inputs()[0].name: seg[np.newaxis, :]}
                    _, _, ovr_r = self.dns_sess.run(None, oi)[0][0]
                    ovr_l.append(self.p_ovr(ovr_r))
                
                ovr = np.mean(ovr_l)
                combined_r = (w_ovr * ovr) 
                
                total_rewards.append(combined_r)
                log_ovrl.append(ovr)
                
            raw_metrics_log = {
                "OVRL": log_ovrl,
            }
                
            return torch.tensor(total_rewards, dtype=torch.float32, device=audio_24k_tensor.device), raw_metrics_log


def main_worker(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cudnn.benchmark = True

    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        
    train_conf = conf["train"]
    grpo_conf = conf["grpo"]
    
    random.seed(train_conf['seed'])
    np.random.seed(train_conf['seed'])
    torch.cuda.manual_seed_all(train_conf['seed'])

    checkpoint_dir = Path(train_conf["checkpoint"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    logger = get_logger(name=(checkpoint_dir / "grpo_trainer.log").as_posix(), file=True)
    
    tb_dir = Path(train_conf.get("tensorboard_dir", str(checkpoint_dir / "tensorboard_logs")))
    tb_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=tb_dir.as_posix())

    logger.info("Initializing CFM base model...")
    nnet = CFM(
        transformer=DiT(**conf['model']['arch'], mel_dim=conf['model']['mel_spec']['n_mel_channels']),
        mel_spec_kwargs=conf['model']['mel_spec']
    ).to(device)
    
    resume_path = train_conf["resume"]
    logger.info(f"Loading base weights: {resume_path}")
    cpt = torch.load(resume_path, map_location="cpu")
    pretrained_dict = cpt["model_state_dict"] if "model_state_dict" in cpt else cpt
    
    # Dimension truncation for text conditioning layers (aligning with base acoustic structure)
    filtered_dict = {k: v for k, v in pretrained_dict.items() if 'text' not in k}
    in_embed_key = 'transformer.input_embed.proj.weight'
    if in_embed_key in filtered_dict and filtered_dict[in_embed_key].shape[1] > 200:
        filtered_dict[in_embed_key] = filtered_dict[in_embed_key][:, :200]
        
    nnet.load_state_dict(filtered_dict, strict=False)

    logger.info("Initializing OmniRewardWrapper...")
    omni_judge = OmniRewardWrapper(conf["reward"]["dnsmos_dir"], conf["reward"]["weights"])
    
    logger.info("Initializing Vocoder...")
    local_vocos_dir = conf['model']['vocoder']['local_path']
    vocoder = Vocos.from_hparams(f"{local_vocos_dir}/config.yaml")
    vocoder.load_state_dict(torch.load(f"{local_vocos_dir}/pytorch_model.bin", map_location="cpu"))
    vocoder = vocoder.to(device)

    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        base_model=nnet,
        vocoder=vocoder,
        dnsmos_model=omni_judge,
        lr=float(conf["optim"]["lr"]),
        group_size=grpo_conf["group_size"],
        ppo_epochs=grpo_conf["ppo_epochs"],
        clip_epsilon=grpo_conf["clip_eps"],
        beta_kl=grpo_conf["beta_kl"],
        sde_a=grpo_conf["sde_a"]
    )

    # Initialize EMA exclusively for trainable LoRA weights
    logger.info("Initializing LoRA EMA (decay=0.9) ...")
    trainable_parameters = list(filter(lambda p: p.requires_grad, trainer.model.parameters()))
    ema = EMAModuleWrapper(trainable_parameters, decay=0.9, update_step_interval=1, device=device)

    grpo_resume_dir = grpo_conf.get("resume_dir", None)
    global_step = 0
    
    # Auto-Resume Logic for RL States
    if grpo_resume_dir and os.path.exists(grpo_resume_dir):
        logger.info(f"Resuming GRPO training from: {grpo_resume_dir}")
        
        lora_path = os.path.join(grpo_resume_dir, "adapter_model.safetensors")
        if os.path.exists(lora_path):
            set_peft_model_state_dict(trainer.model.transformer, load_file(lora_path))
        
        state_path = os.path.join(grpo_resume_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            trainer.optimizer.load_state_dict(state["optimizer"])
            ema.load_state_dict(state["ema"])
            global_step = state["global_step"]
            logger.info(f"Successfully restored global step: {global_step}")
        else:
            match = re.search(r'(?:step|checkpoint)[_-](\d+)', grpo_resume_dir)
            if match:
                global_step = int(match.group(1))
                logger.warning(f"training_state.pt not found. Parsed global step from directory name: {global_step}")
            else:
                logger.warning("training_state.pt not found and parsing failed. Starting from step 0.")
                
        # Forward the scheduler to synchronize the learning rate progression
        for _ in range(global_step):
            trainer.scheduler.step()
        logger.info(f"Restored learning rate aligned to: {trainer.optimizer.param_groups[0]['lr']:.6f}")

    logger.info("Loading training dataset...")
    _, train_loader = make_auto_loader(
        **conf["datasets"]["train"],
        **conf["datasets"]["dataloader_setting"],
        local_rank=local_rank, world_size=1
    )

    logger.info("Starting GRPO optimization loop...")
    
    rl_batch_slice = grpo_conf["rl_batch_slice"]  
    save_period = train_conf["save_period"]
    denoise_steps = grpo_conf["denoise_steps"]
    
    pbar = tqdm(train_loader, desc="GRPO Iteration", dynamic_ncols=True)

    for egs in pbar:
        noisy_mel = egs["noisy_mel"].transpose(-1, -2).to(device)
        clean_mel = egs["label_mel"].transpose(-1, -2).to(device)
        clean_audio = egs['clean_audio'].to(device)

        valid_b = min(rl_batch_slice, noisy_mel.size(0))
        noisy_slice = noisy_mel[:valid_b]
        clean_audio_slice = clean_audio[:valid_b]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            metrics_log, demo_mels = trainer.train_step(cond_batch=noisy_slice, clean_audio=clean_audio_slice, steps=denoise_steps)

        global_step += 1
        ema.step(trainable_parameters, global_step)

        pbar.set_postfix({
            'Policy_Loss': f"{metrics_log.get('Loss/Policy', 0):.6f}",
            'KL_Loss': f"{metrics_log.get('Loss/KL_Div', 0):.6f}",            
            'OVR': f"{metrics_log.get('Raw_OVRL_Batch/Mean', 0):.2f}",
        })

        for k, v in metrics_log.items():
            writer.add_scalar(k, v, global_step)

        if global_step % save_period == 0:
            audio_dir = checkpoint_dir / "audios" / f"step_{global_step}"
            audio_dir.mkdir(exist_ok=True, parents=True)
            logger.info("Performing evaluation and audio saving using current online weights (bypassing EMA for accurate state representation).")

            num_demos = demo_mels.size(0)
            with torch.no_grad():
                ema_demo_mels, _ = trainer.model.sample(
                    cond=noisy_slice[:num_demos], 
                    steps=denoise_steps, 
                    cfg_strength=0.0
                )

                n_mel = noisy_slice[:num_demos].transpose(-1, -2).to(torch.float32)
                c_mel = clean_mel[:valid_b][:num_demos].transpose(-1, -2).to(torch.float32)
                d_mel = ema_demo_mels.transpose(-1, -2).to(torch.float32)
                
                n_wavs = vocoder.decode(n_mel).squeeze(1)
                c_wavs = vocoder.decode(c_mel).squeeze(1)
                d_wavs = vocoder.decode(d_mel).squeeze(1)
                
                # Volume alignment (-25dB) to prevent model from gaming the reward via amplitude scaling
                def norm_wav(w):
                    rms = torch.sqrt(torch.mean(w**2, dim=-1, keepdim=True))
                    return (w * (10**(-25/20) / (rms + 1e-8))).cpu().numpy()
                    
                n_wavs, c_wavs, d_wavs = norm_wav(n_wavs), norm_wav(c_wavs), norm_wav(d_wavs)
        
            for i in range(num_demos):
                writer.add_audio(f"Audio_Step_{global_step}/Sample_{i}_1_Noisy", n_wavs[i], global_step, sample_rate=24000)
                writer.add_audio(f"Audio_Step_{global_step}/Sample_{i}_2_Enhanced", d_wavs[i], global_step, sample_rate=24000)
                writer.add_audio(f"Audio_Step_{global_step}/Sample_{i}_3_Clean", c_wavs[i], global_step, sample_rate=24000)

            save_path = checkpoint_dir / f"grpo_step_{global_step}"
            logger.info(f"Saving LoRA adapter weights to: {save_path}")
            trainer.model.transformer.save_pretrained(save_path.as_posix())

            state_dict = {
                "optimizer": trainer.optimizer.state_dict(),
                "ema": ema.state_dict(),
                "global_step": global_step
            }
            torch.save(state_dict, save_path / "training_state.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowSE GRPO Training Script")
    parser.add_argument("-conf", type=str, required=True, help="Path to GRPO yaml config")
    args = parser.parse_args()
    main_worker(args)