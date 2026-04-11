import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import soundfile as sf
import torchaudio
import librosa

from vocos import Vocos
from model.model_utils import get_tokenizer
from model import DiT, CFM


def get_gpu_memory():
    """Monitor GPU memory usage."""
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    return "N/A"


def normalize(audio, target_level=-25, EPS=np.finfo(float).eps):
    """Normalize the signal to the target level."""
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    return scalar * audio


def run_infer(config_path: str, input_base: str, output_base: str):
    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    
    os.makedirs(output_base, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Initial Memory: {get_gpu_memory()}")

    infer_conf = conf['infer']
    checkpoint_dir = Path(infer_conf["test"]["checkpoint"])
    cpt_fname = checkpoint_dir / infer_conf["test"]["pt_name"]
    
    # Initialize CFM model
    ckpt = torch.load(cpt_fname, map_location=device)
    vocab_char_map, vocab_size = get_tokenizer(conf['model']['tokenizer_path'], conf['model']['tokenizer'])
    
    nnet = CFM(
        transformer=DiT(**conf['model']['arch'], text_num_embeds=vocab_size, mel_dim=conf['model']['mel_spec']['n_mel_channels']),
        mel_spec_kwargs=conf['model']['mel_spec'], 
        vocab_char_map=vocab_char_map
    ).eval().to(device)
    nnet.load_state_dict(ckpt["model_state_dict"])
    
    # Initialize Vocos Vocoder
    vocoder_path = conf['model']['vocoder']['local_path']
    vocoder = Vocos.from_hparams(f"{vocoder_path}/config.yaml")
    vocoder.load_state_dict(torch.load(f"{vocoder_path}/pytorch_model.bin", map_location="cpu"))
    vocoder = vocoder.eval().to(device)

    input_base_path = Path(input_base)
    output_base_path = Path(output_base)
    
    target_dirs = [
        "synthetic/no_reverb/noisy",
    ]

    resampler_16k_to_24k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000).to(device)

    with torch.no_grad():
        for sub_dir in target_dirs:
            current_in_dir = input_base_path / sub_dir
            if not current_in_dir.exists():
                print(f"[Skip] Directory does not exist: {current_in_dir}")
                continue

            wav_files = list(current_in_dir.glob("**/*.wav"))
            print(f"Processing {sub_dir} ({len(wav_files)} files)")

            for wav_path in tqdm(wav_files):
                mix, sr = torchaudio.load(str(wav_path))
                mix = mix.to(device)
                
                if sr != 24000:
                    mix = resampler_16k_to_24k(mix)

                output, _ = nnet.sample(cond=mix, text=[" "], drop_text=True) 
                
                output = output.transpose(-1, -2).to(torch.float32)
                generated_wave = vocoder.decode(output).squeeze().cpu().numpy() 

                generated_wave = librosa.resample(normalize(generated_wave), orig_sr=24000, target_sr=16000)

                rel_path = wav_path.relative_to(input_base_path)
                save_path = output_base_path / rel_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                sf.write(str(save_path), generated_wave, 16000)
    
    print(f"Final peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowSE Inference Script")
    parser.add_argument("--config", type=str, default="./config/train.yaml", help="Path to config file")
    parser.add_argument("--input_dir", type=str, default="./datasets/test_set", help="Base directory for input noisy audio")
    parser.add_argument("--output_dir", type=str, default="./outputs/flowSE_infer_16k", help="Base directory to save generated audio")
    args = parser.parse_args()

    run_infer(args.config, args.input_dir, args.output_dir)