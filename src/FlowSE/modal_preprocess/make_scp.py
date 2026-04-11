import os
import glob
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path


def process_file(file_path):
    """
    读取音频文件时长，并将长音频逻辑切分为最大 10 秒的片段。
    输出格式: [文件路径] [起始时间] [结束时间] [片段时长]
    """
    try:
        info = sf.info(file_path)
        duration = info.duration
        
        valid_lines = []
        chunk_sec = 10.0
        
        if duration >= 1.0:
            if duration > chunk_sec:
                num_chunks = int(duration // chunk_sec)
                for i in range(num_chunks):
                    start = i * chunk_sec
                    end = start + chunk_sec
                    valid_lines.append(f"{file_path} {start:.4f} {end:.4f} {chunk_sec:.4f}\n")
                
                # 处理剩余不足一个 Chunk 的部分
                tail = duration % chunk_sec
                if tail >= 1.0:
                    valid_lines.append(f"{file_path} {duration-tail:.4f} {duration:.4f} {tail:.4f}\n")
            else:
                valid_lines.append(f"{file_path} 0.0000 {duration:.4f} {duration:.4f}\n")
                
        return valid_lines 
    except Exception:
        return None


def generate_scp(dataset_dirs, output_scp_path, num_workers=16):
    """多进程遍历目录结构并生成数据流 SCP 列表"""
    print(f"Scanning directories for: {output_scp_path}")
    
    all_files = []
    for d in dataset_dirs:
        all_files.extend(glob.glob(os.path.join(d, '**', '*.wav'), recursive=True))
        all_files.extend(glob.glob(os.path.join(d, '**', '*.flac'), recursive=True))
        all_files.extend(glob.glob(os.path.join(d, '**', '*.opus'), recursive=True))
    
    print(f"Found {len(all_files)} audio files. Parsing metadata via multiprocessing...")
    
    valid_lines = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing duration"):
            result = future.result()
            if result is not None:
                valid_lines.extend(result)
                
    output_path = Path(output_scp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_scp_path, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
        
    print(f"Generation completed. {len(valid_lines)} valid segments saved to {output_scp_path}\n")


def main():
    clean_dirs = [
        "./data/gigaspeech", 
        "./data/wenetspeech_extracted/audio/train",
        "./data/dns3/datasets/clean/read_speech",
        "./data/dns3/datasets/clean/emotional_speech",
        "./data/voicebank/train_clean"
    ]
    generate_scp(clean_dirs, "./data/train_clean_en_cn.scp")
    
    noise_dirs = [ 
        "./data/dns3/datasets/noise",
        "./data/dns3/demand",
        "./data/noise"
    ]
    generate_scp(noise_dirs, "./data/train_noise_en_cn.scp")
    
    rir_dirs = [
        "./data/dns3/datasets/impulse_responses"
    ]
    generate_scp(rir_dirs, "./data/train_rir.scp")


if __name__ == "__main__":

    main()