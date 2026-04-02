import os
import glob
import re
import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import onnxruntime as ort
import soundfile as sf
import wespeaker
import jiwer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import PeftModel

from vocos import Vocos
from model import DiT, CFM


def normalize(audio, target_level=-25, EPS=np.finfo(float).eps):
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    return scalar * audio


class GPUDNSMOSFeature(torch.nn.Module):
    def __init__(self, sr=16000, n_mels=120, frame_size=320, hop_length=160):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=frame_size+1, win_length=frame_size,
            hop_length=hop_length, n_mels=n_mels, center=True, pad_mode="reflect"
        )
        self.resampler = torchaudio.transforms.Resample(24000, sr)

    @torch.no_grad()
    def forward(self, audio_24k):
        audio_16k = self.resampler(audio_24k)
        mel = self.mel_transform(audio_16k)
        mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        mel = mel - torch.max(mel)
        mel = (mel + 40.0) / 40.0
        return mel.transpose(1, 2) 


class FastDNSMOS:
    def __init__(self, primary_path, p808_path):
        providers = ['CUDAExecutionProvider']
        self.sess = ort.InferenceSession(primary_path, providers=providers)
        self.p808_sess = ort.InferenceSession(p808_path, providers=providers)
        self.feature_extractor = GPUDNSMOSFeature().cuda()
        self.p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
        self.p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
        self.p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

    def score(self, audio_24k_tensor):
        audio_16k = self.feature_extractor.resampler(audio_24k_tensor)
        aud = audio_16k.squeeze(0).cpu().numpy().astype(np.float32) 
        len_samples = int(9.01 * 16000)
        
        while len(aud) < len_samples:
            aud = np.append(aud, aud)
        num_hops = int(np.floor(len(aud) / 16000) - 9.01) + 1
        if num_hops < 1: 
            num_hops = 1

        sig_list, bak_list, ovr_list, p808_list = [], [], [], []
        for idx in range(num_hops):
            start = int(idx * 16000)
            seg = aud[start:start + len_samples]
            if len(seg) < len_samples:
                seg = np.pad(seg, (0, len_samples - len(seg)), 'wrap')

            oi = {self.sess.get_inputs()[0].name: seg[np.newaxis, :]}
            sig_r, bak_r, ovr_r = self.sess.run(None, oi)[0][0]
            sig, bak, ovr = self.p_sig(sig_r), self.p_bak(bak_r), self.p_ovr(ovr_r)
            
            seg_p808_tensor = torch.from_numpy(seg[:-160]).unsqueeze(0).cuda()
            mel = self.feature_extractor.mel_transform(seg_p808_tensor)
            mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
            mel = mel - torch.max(mel)
            mel = (mel + 40.0) / 40.0
            
            p808_oi = {self.p808_sess.get_inputs()[0].name: mel.transpose(1, 2).cpu().numpy().astype(np.float32)}
            p808_mos = self.p808_sess.run(None, p808_oi)[0][0][0]
            
            sig_list.append(sig); bak_list.append(bak)
            ovr_list.append(ovr); p808_list.append(p808_mos)

        return np.mean(sig_list), np.mean(bak_list), np.mean(ovr_list), np.mean(p808_list)


def run_batch_eval(config_path, lora_dir, base_ckpt_path, dns3_dir, onnx_dir, gt_feature_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Model Initialization --------
    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    
    nnet = CFM(
        transformer=DiT(**conf['model']['arch'], mel_dim=conf['model']['mel_spec']['n_mel_channels']),
        mel_spec_kwargs=conf['model']['mel_spec']
    ).to(device)
    
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu")
    state_dict_key = "ema_state_dict" if "ema_state_dict" in base_ckpt else "model_state_dict"
    pretrained_dict = base_ckpt[state_dict_key]
    
    embed_keys = ['transformer.input_embed.proj.weight', 'module.transformer.input_embed.proj.weight']
    for k in embed_keys:
        if k in pretrained_dict:
            old_weight = pretrained_dict[k]
            if old_weight.shape[1] > 200:
                pretrained_dict[k] = old_weight[:, :200]

    nnet.load_state_dict(pretrained_dict, strict=False)
    nnet = nnet.to(device).eval()

    local_vocos_dir = conf['model']['vocoder']['local_path']
    vocoder = Vocos.from_hparams(f"{local_vocos_dir}/config.yaml")
    vocoder.load_state_dict(torch.load(f"{local_vocos_dir}/pytorch_model.bin", map_location="cpu"))
    vocoder = vocoder.to(device).eval()

    dnsmos = FastDNSMOS(f"{onnx_dir}/sig_bak_ovr.onnx", f"{onnx_dir}/model_v8.onnx")
    spk_model = wespeaker.load_model('english')
    spk_model.set_device('cpu')
    
    model_id = "openai/whisper-large-v3-turbo"
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, 
        use_safetensors=True, attn_implementation="sdpa"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    text_normalizer = jiwer.RemovePunctuation()
    resampler_24_to_16 = torchaudio.transforms.Resample(24000, 16000).to(device)

    # -------- Dataset and Features Loading --------
    if not Path(gt_feature_path).exists():
        raise FileNotFoundError(f"Ground truth features missing: {gt_feature_path}")
    gt_features = torch.load(gt_feature_path, map_location="cpu")
    gt_keys = list(gt_features.keys())

    test_files = []
    base_path = Path(dns3_dir)
    dir_path = base_path / "synthetic/no_reverb/noisy"
    
    if dir_path.exists():
        for p in dir_path.glob("*.wav"):
            file_key = None
            core_match = re.search(r'(fileid_\d+)', p.name)
            if core_match:
                core_id = core_match.group(1)
                target_suffix = f"{core_id}.wav" 
                for k in gt_keys:
                    if k.endswith(target_suffix):
                        file_key = k
                        break
            
            if not file_key and p.name in gt_features: 
                file_key = p.name
            if file_key and file_key in gt_features:
                test_files.append({"path": p, "cat": "No_Reverb", "key": file_key})

    lora_bins = glob.glob(f"{lora_dir}/**/adapter_model.bin", recursive=True)
    if not lora_bins:  
        lora_bins = glob.glob(f"{lora_dir}/*.bin") + glob.glob(f"{lora_dir}/**/*.safetensors", recursive=True)

    def extract_num(path):
        match = re.search(r'(?:step|checkpoint)[_-](\d+)', path)
        return int(match.group(1)) if match else 0

    # 在这里选择需要测的步数 
    lora_bins.sort(key=extract_num)
    ckpts_to_eval = lora_bins[-1:] if lora_bins else []
    
    print(f"Total evaluation samples: {len(test_files)}. Target LoRA checkpoint: {ckpts_to_eval}")

    memory_dataset = []
    resampler_cache = {}
    for item in tqdm(test_files, desc="Preloading Audio"):
        wav, sr = torchaudio.load(str(item["path"]))
        if sr not in resampler_cache: 
            resampler_cache[sr] = torchaudio.transforms.Resample(sr, 24000)
        wav_24k = resampler_cache[sr](wav)
        memory_dataset.append({
            "wav": wav_24k, "length": wav_24k.shape[1], 
            "category": item["cat"], "key": item["key"]
        })

    # -------- Evaluation Loop --------
    all_results = []
    batch_size = 16 
    
    for bin_path in ckpts_to_eval:
        match = re.search(r'(?:step|checkpoint)[_-](\d+)', bin_path)
        step_num = match.group(1) if match else Path(bin_path).parent.name
        print(f"\nEvaluating LoRA Checkpoint: {step_num}")
        
        lora_dir_path = str(Path(bin_path).parent)
        
        try:
            if hasattr(nnet.transformer, "peft_config"):
                nnet.transformer.delete_adapter("default")
                nnet.transformer.load_adapter(lora_dir_path, adapter_name="default")
            else:
                nnet.transformer = PeftModel.from_pretrained(
                    nnet.transformer, 
                    lora_dir_path, 
                    adapter_name="default", 
                    is_trainable=False
                )
            
            nnet = nnet.to(device).eval()
        except Exception as e:
            print(f"LoRA loading failed: {e}")
            continue

        cat_scores = {"No_Reverb": {"SIG":[], "BAK":[], "OVRL":[], "P808":[], "SPK":[], "WER":[]}}

        for i in tqdm(range(0, len(memory_dataset), batch_size), desc=f"Eval LoRA {step_num}"):
            batch_data = memory_dataset[i:i+batch_size]
            wavs = [item["wav"].to(device) for item in batch_data]
            lengths = [item["length"] for item in batch_data]
            
            max_len = max(lengths)
            padded_wavs = torch.cat([F.pad(w, (0, max_len - w.shape[1])) for w in wavs], dim=0)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs, _ = nnet.sample(cond=padded_wavs, steps=12, cfg_strength=0.0)
            outputs = outputs.transpose(-1, -2).to(torch.float32)
            gen_waves_24k = vocoder.decode(outputs) 

            for b_idx in range(len(batch_data)):
                cat_name = batch_data[b_idx]["category"]
                file_key = batch_data[b_idx]["key"]
                valid_len = lengths[b_idx]
                
                gen_wav_24k = gen_waves_24k[b_idx:b_idx+1, :valid_len] 
                gen_wav_16k = resampler_24_to_16(gen_wav_24k)
                gen_wav_16k_cpu = gen_wav_16k.squeeze(0).cpu().numpy()
                
                dns_scores = dnsmos.score(normalize(gen_wav_24k))
                cat_scores[cat_name]["SIG"].append(dns_scores[0])
                cat_scores[cat_name]["BAK"].append(dns_scores[1])
                cat_scores[cat_name]["OVRL"].append(dns_scores[2])
                cat_scores[cat_name]["P808"].append(dns_scores[3])
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_name = f.name
                sf.write(tmp_name, gen_wav_16k_cpu, 16000)
                
                with torch.no_grad():
                    gen_emb = spk_model.extract_embedding(tmp_name)
                os.remove(tmp_name)
                
                gen_emb_tensor = gen_emb.squeeze().to(device)
                gt_emb = gt_features[file_key]["emb"].to(device)
                sim = F.cosine_similarity(gen_emb_tensor, gt_emb, dim=0).item()
                cat_scores[cat_name]["SPK"].append(sim)
                
                with torch.no_grad():
                    inputs = processor(gen_wav_16k_cpu, sampling_rate=16000, return_tensors="pt")
                    input_features = inputs.input_features.to(device)
                    predicted_ids = whisper_model.generate(
                        inputs=input_features, language="english", return_timestamps=True, num_beams=5         
                    )
                    raw_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                gen_text_norm = text_normalizer(raw_text.lower().strip())
                gt_text_norm = gt_features[file_key]["text"]
                
                if len(gt_text_norm.strip()) == 0:
                    wer_score = 0.0 if len(gen_text_norm.strip()) == 0 else 1.0
                else:
                    try:
                        wer_score = jiwer.wer(gt_text_norm, gen_text_norm)
                    except ValueError:
                        wer_score = 1.0 
                cat_scores[cat_name]["WER"].append(wer_score)
                
        step_summary = {"Step": f"LoRA_{step_num}"}
        print(f"\n[LoRA {step_num} Evaluation Results]:")
        
        for cat in ["No_Reverb"]:
            scores = cat_scores[cat]
            if scores["WER"]: 
                step_summary[f"{cat}_SIG"] = round(np.mean(scores["SIG"]), 3)
                step_summary[f"{cat}_BAK"] = round(np.mean(scores["BAK"]), 3)
                step_summary[f"{cat}_OVRL"] = round(np.mean(scores["OVRL"]), 3)
                step_summary[f"{cat}_P808"] = round(np.mean(scores["P808"]), 3)
                step_summary[f"{cat}_SPK"] = round(np.mean(scores["SPK"]), 3)
                step_summary[f"{cat}_WER"] = round(np.mean(scores["WER"]), 4)
                
                print(f"  DNSMOS | SIG: {step_summary[f'{cat}_SIG']:.3f} | BAK: {step_summary[f'{cat}_BAK']:.3f} | OVRL: {step_summary[f'{cat}_OVRL']:.3f}")
                print(f"  Metrics | SPK: {step_summary[f'{cat}_SPK']:.3f} | WER: {step_summary[f'{cat}_WER']*100:.2f}%")
                
        all_results.append(step_summary)
        
        df = pd.DataFrame(all_results)
        save_csv_path = f"{lora_dir}/grpo_metrics_history.csv"
        df.to_csv(save_csv_path, index=False)

    print(f"\nEvaluation completed. Results saved to: {save_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA GRPO Checkpoints Evaluation")
    parser.add_argument("--config", type=str, default="./config/my_grpo.yaml", help="Path to config file")
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory containing LoRA checkpoints")
    parser.add_argument("--base_ckpt_path", type=str, required=True, help="Path to the base model checkpoint")
    parser.add_argument("--dns3_dir", type=str, default="./datasets/test_set", help="Path to the test dataset")
    parser.add_argument("--onnx_dir", type=str, default="./DNSMOS/", help="Path to DNSMOS ONNX models")
    parser.add_argument("--gt_feature_path", type=str, default="./datasets/test_set/clean_gt_features.pt", help="Path to ground truth features")
    
    args = parser.parse_args()
    run_batch_eval(args.config, args.lora_dir, args.base_ckpt_path, args.dns3_dir, args.onnx_dir, args.gt_feature_path)