import os
import glob
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import librosa

calculator = None


class CPU_DNSMOS:
    """DNSMOS CPU Inference Class."""
    def __init__(self, primary_path, p808_path):
        import onnxruntime as ort
        
        # 限制每个 ONNX 会话的线程数，防止多进程互相争抢 CPU 资源导致死锁或上下文切换过载
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 1
        sess_opt.inter_op_num_threads = 1
        
        self.sess = ort.InferenceSession(primary_path, sess_options=sess_opt, providers=['CPUExecutionProvider'])
        self.p808_sess = ort.InferenceSession(p808_path, sess_options=sess_opt, providers=['CPUExecutionProvider'])
        
        self.p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
        self.p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
        self.p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

    def score(self, audio_16k):
        # 1. 主模型推理 (输入维度要求: [1, 144160])
        oi = {self.sess.get_inputs()[0].name: audio_16k[np.newaxis, :]}
        sig_r, bak_r, ovr_r = self.sess.run(None, oi)[0][0]
        sig, bak, ovr = self.p_sig(sig_r), self.p_bak(bak_r), self.p_ovr(ovr_r)

        # 2. 提取 Mel 频谱供 P808 模型使用 (舍弃最后 160 个点)
        seg_p808 = audio_16k[:-160]
        mel = librosa.feature.melspectrogram(
            y=seg_p808, sr=16000, n_fft=321, hop_length=160, n_mels=120, center=True, pad_mode="reflect"
        )
        mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel = mel - np.max(mel)
        mel = (mel + 40.0) / 40.0
        mel = mel.T[np.newaxis, :, :] 

        # 3. P808 模型推理
        p808_oi = {self.p808_sess.get_inputs()[0].name: mel.astype('float32')}
        p808_mos = self.p808_sess.run(None, p808_oi)[0][0][0]

        return sig, bak, ovr, p808_mos


def init_worker(primary_path, p808_path):
    global calculator
    calculator = CPU_DNSMOS(primary_path, p808_path)


def process_file(file_path):
    """
    自适应采样与打分逻辑：
    - 时长 < 9.01s: 全量补齐打分
    - 时长 9.01s ~ 30s: 截取中间的 9 秒打分
    - 时长 > 30s: 采用 15%, 50%, 85% 三点采样法，分别打分后取平均
    """
    global calculator
    
    try:
        info = sf.info(file_path)
        sr = info.samplerate
        duration = info.duration

        if duration < 1.0: 
            return None 

        sample_starts = []
        if duration < 9.01:
            sample_starts.append(0.0)
        elif duration < 30.0:
            sample_starts.append((duration - 9.01) / 2)
        else:
            sample_starts = [duration * 0.15, duration * 0.50, duration * 0.85]

        sigs, baks, ovrs, p808s = [], [], [], []
        target_len = int(9.01 * 16000)

        for start_sec in sample_starts:
            start_frame = int(start_sec * sr)
            frames_to_read = int(9.01 * sr)
            
            audio, _ = sf.read(file_path, start=start_frame, frames=frames_to_read, dtype='float32', always_2d=True)
            
            audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio.squeeze(1)
            if len(audio) == 0: 
                continue

            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            rms = np.sqrt(np.mean(audio**2) + 1e-9)
            audio = audio * (10 ** (-25 / 20)) / rms

            while len(audio) < target_len:
                audio = np.append(audio, audio)
            audio = audio[:target_len]

            sig, bak, ovr, p808 = calculator.score(audio)
            sigs.append(sig)
            baks.append(bak)
            ovrs.append(ovr)
            p808s.append(p808)
            
        if not sigs: 
            return None

        return {
            "Filepath": file_path,
            "Duration_sec": round(duration, 2),
            "SIG": round(float(np.mean(sigs)), 3),
            "BAK": round(float(np.mean(baks)), 3),
            "OVRL": round(float(np.mean(ovrs)), 3),
            "P808": round(float(np.mean(p808s)), 3),
            "OVRL_MIN": round(float(np.min(ovrs)), 3) 
        }
    except Exception as e:
        return None


def run_dataset_curation(clean_dirs, onnx_primary, onnx_p808, output_csv, num_workers=8):
    all_files = []
    for d in clean_dirs:
        for ext in ['*.wav', '*.flac', '*.opus']:
            all_files.extend(glob.glob(os.path.join(d, '**', ext), recursive=True))

    # 断点续传机制：读取历史记录，过滤已处理文件
    processed_files = set()
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv)
            if 'Filepath' in df_existing.columns:
                processed_files = set(df_existing['Filepath'].tolist())
                print(f"Loaded existing checkpoint. Skipped {len(processed_files)} processed records.")
        except Exception as e:
            print(f"[Warning] Failed to read existing CSV checkpoint: {e}")
            
    pending_files = [f for f in all_files if f not in processed_files]
    print(f"Total files: {len(all_files)}. Pending for evaluation: {len(pending_files)}")
    
    if not pending_files:
        print("All files have been evaluated. Process complete.")
        return
    
    if not os.path.exists(output_csv):
        cols = ["Filepath", "Duration_sec", "SIG", "BAK", "OVRL", "P808", "OVRL_MIN"]
        pd.DataFrame(columns=cols).to_csv(output_csv, index=False)

    results_buffer = []
    save_interval = 1000 

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(onnx_primary, onnx_p808)) as executor:
        for res in tqdm(executor.map(process_file, pending_files, chunksize=50), total=len(pending_files), desc="Data Curation"):
            if res:
                results_buffer.append(res)
                
                if len(results_buffer) >= save_interval:
                    df_buffer = pd.DataFrame(results_buffer)
                    df_buffer.to_csv(output_csv, mode='a', header=False, index=False)
                    results_buffer.clear()

    if results_buffer:
        df_buffer = pd.DataFrame(results_buffer)
        df_buffer.to_csv(output_csv, mode='a', header=False, index=False)
        
    print(f"Data curation complete. Report saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNSMOS Batch Evaluation Pipeline")
    parser.add_argument("--onnx_dir", type=str, default="./DNSMOS", help="Directory containing ONNX models")
    parser.add_argument("--output_csv", type=str, default="./data/clean_dataset_dnsmos_scores.csv", help="Path to save the evaluation report")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU parallel workers")
    args = parser.parse_args()

    # 替换为实际路径
    target_dirs = [
        "./data/gigaspeech", 
        "./data/wenetspeech_extracted/audio/train",
        "./data/dns3/datasets/clean/read_speech",
        "./data/dns3/datasets/clean/emotional_speech",
        "./data/voicebank/train_clean"
    ]
    
    primary_model_path = os.path.join(args.onnx_dir, "sig_bak_ovr.onnx")
    p808_model_path = os.path.join(args.onnx_dir, "model_v8.onnx")
    
    run_dataset_curation(target_dirs, primary_model_path, p808_model_path, args.output_csv, args.num_workers)