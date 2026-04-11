import os
import math
import random
import ctypes
import gc
import json
import functools
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import scipy.signal as sps
import torch
import torch.nn.functional as F
import torchaudio
import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Sampler
import torch.distributed as dist
from tqdm import tqdm

import sys
sys.path.append("../")
from model.modules import MelSpec

EPS = np.finfo(float).eps


class IndexOnlyDataset(torch.utils.data.Dataset):
    """
    轻量级数据集占位符，避免在主进程中读取实际音频数据，
    仅返回索引，实际的数据读取通过线程池在 collate_fn 中完成。
    """
    def __init__(self, length):
        self.length = length
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        return idx


def is_clipped(audio, clipping_threshold=0.99):
    return torch.any(torch.abs(audio) > clipping_threshold)


def normalize(audio, target_level=-25):
    """Normalize the signal to the target level using PyTorch"""
    rms = torch.sqrt(torch.mean(audio**2))
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


@functools.lru_cache(maxsize=8)
def get_cached_resampler(orig_freq, new_freq):
    """使用 LRU Cache 缓存重采样器，避免每次读取时重复初始化"""
    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)


def get_firstchannel_read(data_info, fs=16000):
    """
    使用 soundfile 进行线程安全的音频读取与切片。
    """
    path = data_info["inputs"]
    start_sec = data_info.get("start", 0.0)
    duration_sec = data_info.get("duration", -1.0)
    
    try:
        if duration_sec != -1.0:
            info = sf.info(path)
            orig_sr = info.samplerate
            frame_offset = int(start_sec * orig_sr)
            num_frames = int(duration_sec * orig_sr)
            wave_numpy, sr = sf.read(path, start=frame_offset, frames=num_frames, dtype='float32')
        else:
            wave_numpy, sr = sf.read(path, dtype='float32')
            
        if wave_numpy.ndim == 1:
            wave_data = torch.from_numpy(wave_numpy).unsqueeze(0)
        else:
            wave_data = torch.from_numpy(wave_numpy).t()
            
        if sr != fs:
            resampler = get_cached_resampler(sr, fs)
            wave_data = resampler(wave_data)
            
        return wave_data[0, :]
        
    except Exception as e:
        print(f"[Warning] Audio read failed, fallback to silence. Path: {path}, Error: {e}")
        fallback_length = int(fs * (duration_sec if duration_sec > 0 else 3.0))
        return torch.zeros(fallback_length)


def audioread(path, fs=16000):
    try:
        wave_numpy, sr = sf.read(path, dtype='float32')
        if wave_numpy.ndim == 1:
            wave_data = torch.from_numpy(wave_numpy).unsqueeze(0)
        else:
            wave_data = torch.from_numpy(wave_numpy).t()
            
        if sr != fs:
            resampler = get_cached_resampler(sr, fs)
            wave_data = resampler(wave_data)
            
        return wave_data.transpose(0, 1)
    except Exception as e:
        print(f"[Warning] audioread failed: {path}, Error: {e}")
        return torch.zeros(fs)


def add_reverb(cln_wav, rir_wav):
    wav_tgt = sps.oaconvolve(cln_wav.numpy(), rir_wav.numpy())
    wav_tgt = wav_tgt[: cln_wav.shape[0]]
    return torch.tensor(wav_tgt)


def db2num(y):
    return torch.pow(10.0, y / 20.0)


def parse_scp(scp, path_list, test=-1, split_token=" "):
    with open(scp) as fid:
        if not test == -1:
            total = 500
            count = 0
        for line in fid:
            if not test == -1:
                if count > total:
                    break
                count += 1
            tmp = line.strip().split(split_token)
            if len(tmp) == 4:
                path_list.append({
                    "inputs": tmp[0], 
                    "start": float(tmp[1]), 
                    "end": float(tmp[2]), 
                    "duration": float(tmp[3])
                })
            elif len(tmp) == 2:
                path_list.append({
                    "inputs": tmp[0], 
                    "start": 0.0, 
                    "end": float(tmp[1]), 
                    "duration": float(tmp[1])
                })


def pad(audio, chunk_length, randstate):
    audio_length = audio.shape[0]

    if chunk_length > audio_length:
        st = randstate.randint(chunk_length + 1 - audio_length)
        audio_t = torch.zeros(chunk_length, dtype=audio.dtype)
        audio_t[st : st + audio_length] = audio
        audio = audio_t
    elif chunk_length < audio_length:
        st = randstate.randint(audio_length + 1 - chunk_length)
        audio = audio[st : st + chunk_length]
    return audio


def generate_data_one_noise(
    clean, noise, snr, scale, target_level=-25, clipping_threshold=0.99
):
    clean = clean / (torch.max(torch.abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = torch.sqrt(torch.mean(clean**2))

    noise = noise / (torch.max(torch.abs(noise)) + EPS)
    noise = normalize(noise, target_level)
    rmsnoise = torch.sqrt(torch.mean(noise**2))

    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    noisyspeech = clean + noisenewlevel
    noisy_rms_level = scale

    rmsnoisy = torch.sqrt(torch.mean(noisyspeech**2))
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy

    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = torch.max(torch.abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel

    return noisyspeech, clean


def generate_data_two_noise(
    clean, noise1, noise2, snr_noise1, snr_noise2, scale, target_level=-25, clipping_threshold=0.99,
):
    clean = clean / (torch.max(torch.abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = torch.sqrt(torch.mean(clean**2))

    noise1 = noise1 / (torch.max(torch.abs(noise1)) + EPS)
    noise1 = normalize(noise1, target_level)
    rmsnoise1 = torch.sqrt(torch.mean(noise1**2))

    noise2 = noise2 / (torch.max(torch.abs(noise2)) + EPS)
    noise2 = normalize(noise2, target_level)
    rmsnoise2 = torch.sqrt(torch.mean(noise2**2))

    noise1scalar = rmsclean / (10 ** (snr_noise1 / 20)) / (rmsnoise1 + EPS)
    noise1newlevel = noise1 * noise1scalar
    noise2scalar = rmsclean / (10 ** (snr_noise2 / 20)) / (rmsnoise2 + EPS)
    noise2newlevel = noise2 * noise2scalar

    noisyspeech = clean + noise1newlevel + noise2newlevel
    noisy_rms_level = scale

    rmsnoisy = torch.sqrt(torch.mean(noisyspeech**2))
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy

    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = torch.max(torch.abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel

    return noisyspeech, clean


def generate_reverdata_one_reverb_noise(clean, noise1, rir, snr, scale):
    clean_rir = rir[:, 0]
    noise_rir = rir[:, 1] if rir.shape[1] > 1 else rir[:, 0] 
    
    clean = add_reverb(clean, clean_rir)
    noise1 = add_reverb(noise1, noise_rir)
    noisyspeech, clean = generate_data_one_noise(clean, noise1, snr, scale)
    return noisyspeech, clean


def generate_reverdata_one_noise(clean, noise1, rir, snr, scale):
    clean_rir = rir[:, 0]
    clean = add_reverb(clean, clean_rir)
    noisyspeech, clean = generate_data_one_noise(clean, noise1, snr, scale)
    return noisyspeech, clean


def generate_reverdata_two_reverb_noise(clean, noise1, noise2, rir, snr1, snr2, scale):
    clean_rir = rir[:, 0]
    noise1_rir = rir[:, 1] if rir.shape[1] > 1 else rir[:, 0]
    noise2_rir = rir[:, 2] if rir.shape[1] > 2 else rir[:, 0]
    
    clean = add_reverb(clean, clean_rir)
    noise1 = add_reverb(noise1, noise1_rir)
    noise2 = add_reverb(noise2, noise2_rir)
    noisyspeech, clean = generate_data_two_noise(clean, noise1, noise2, snr1, snr2, scale)
    return noisyspeech, clean


def generate_reverdata_one_reverb_noise_one_noise(
    clean, noise1, noise2, rir, snr1, snr2, scale
):
    clean_rir = rir[:, 0]
    noise1_rir = rir[:, 1] if rir.shape[1] > 1 else rir[:, 0]
    
    clean = add_reverb(clean, clean_rir)
    noise1 = add_reverb(noise1, noise1_rir)
    noisyspeech, clean = generate_data_two_noise(clean, noise1, noise2, snr1, snr2, scale)
    return noisyspeech, clean


class AutoDataset(Dataset):
    def __init__(
        self,
        clean_scp,
        regular_noise_scp,
        rir_scp,
        text_scp,
        repeat=1,
        num_workers=40,
        sample_rate=16000,
        probability=None,
        snr_ranges=None,
        scale_ranges=None,
    ):
        super(AutoDataset, self).__init__()

        self.probaction = list(probability.keys())
        self.probvalues = list(probability.values())
        self.snr_ranges = snr_ranges
        self.scale_ranges = scale_ranges

        self.clean_list = []
        self.regular_noise_list = []
        self.rir_list = []
        
        parse_scp(clean_scp, self.clean_list)
        parse_scp(regular_noise_scp, self.regular_noise_list)
        parse_scp(rir_scp, self.rir_list)

        self.len_clean = len(self.clean_list)
        self.len_regular_noise = len(self.regular_noise_list)
        self.len_rir = len(self.rir_list)

        self.index = list(self.clean_list) * repeat

        self.mel_spectrogram = MelSpec(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type='vocos',
        )
        self.randstates = [np.random.RandomState(idx) for idx in range(3000)]
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)

    def __len__(self):
        return len(self.index)

    def name(self, path):
        return os.path.splitext(os.path.basename(path))[0]

    def __next_probaiblity__(self):
        action = random.choices(self.probaction, self.probvalues)[0]
        return action

    def __select_rand_number__(self, probabilities, randstate):
        ranges = list(probabilities.keys())
        probs = list(probabilities.values())

        selected_range = randstate.choice(ranges, p=probs)
        start, end = map(int, selected_range.split("_to_"))

        rand_number = randstate.uniform(start, end)
        return rand_number

    def get_frame_len(self, index):
        """强制上限 10 秒，防止 batch_sampler 被异常长音频影响性能分布"""
        duration = min(self.index[index]["duration"], 10.0)
        return duration * 24000 / 256

    def __getitem__(self, index):
        data_info = self.index[index]
        clean_path = data_info["inputs"]

        clean = get_firstchannel_read(data_info)
        chunk_length = clean.shape[-1]
      
        randstate = self.randstates[(index + 11) % 3000]

        idx_noise1 = randstate.randint(0, self.len_regular_noise)
        idx_noise2 = randstate.randint(0, self.len_regular_noise)
        while idx_noise2 == idx_noise1:
            idx_noise2 = randstate.randint(0, self.len_regular_noise)
        idx_rir = randstate.randint(0, self.len_rir)

        noise1_info = self.regular_noise_list[idx_noise1]
        noise2_info = self.regular_noise_list[idx_noise2]
        rir_info = self.rir_list[idx_rir]

        snr1 = self.__select_rand_number__(self.snr_ranges, randstate)
        snr2 = self.__select_rand_number__(self.snr_ranges, randstate)
        scale = self.__select_rand_number__(self.scale_ranges, randstate)

        choice = self.__next_probaiblity__()
        
        if choice == 'p1':
            noise1 = get_firstchannel_read(noise1_info)
            noise1 = pad(noise1, chunk_length, randstate)
            inputs, labels = generate_data_one_noise(clean, noise1, snr1, scale)
        elif choice == 'p2':
            noise1 = get_firstchannel_read(noise1_info)
            noise2 = get_firstchannel_read(noise2_info)
            noise1 = pad(noise1, chunk_length, randstate)
            noise2 = pad(noise2, chunk_length, randstate)
            inputs, labels = generate_data_two_noise(clean, noise1, noise2, snr1, snr2, scale)
        elif choice == 'p3':
            noise1 = get_firstchannel_read(noise1_info)
            noise1 = pad(noise1, chunk_length, randstate)
            rir = audioread(rir_info["inputs"]) 
            inputs, labels = generate_reverdata_one_reverb_noise(clean, noise1, rir, snr1, scale)
        elif choice == 'p4':
            noise1 = get_firstchannel_read(noise1_info)
            noise2 = get_firstchannel_read(noise2_info)
            noise1 = pad(noise1, chunk_length, randstate)
            noise2 = pad(noise2, chunk_length, randstate)
            rir = audioread(rir_info["inputs"])
            inputs, labels = generate_reverdata_two_reverb_noise(clean, noise1, noise2, rir, snr1, snr2, scale)
        elif choice == 'p5':
            noise1 = get_firstchannel_read(noise1_info)
            noise1 = pad(noise1, chunk_length, randstate)
            rir = audioread(rir_info["inputs"])
            inputs, labels = generate_reverdata_one_noise(clean, noise1, rir, snr1, scale)
        else: 
            noise1 = get_firstchannel_read(noise1_info)
            noise2 = get_firstchannel_read(noise2_info)
            noise1 = pad(noise1, chunk_length, randstate)
            noise2 = pad(noise2, chunk_length, randstate)
            rir = audioread(rir_info["inputs"])
            inputs, labels = generate_reverdata_one_reverb_noise_one_noise(clean, noise1, noise2, rir, snr1, snr2, scale)

        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)
        
        inputs = self.resampler(inputs)
        labels = self.resampler(labels)

        noisy_mel_spec = self.mel_spectrogram(inputs).squeeze(0) 
        label_mel_spec = self.mel_spectrogram(labels).squeeze(0)

        egs = {
            "noisy": inputs, 
            "clean": labels,
            "label_mel_spec": label_mel_spec, 
            "noisy_mel_spec": noisy_mel_spec, 
            "label_path": clean_path
        }
        return egs


def worker(target_list, result_list, start, end, chunk_length, sample_rate):
    for item in target_list[start:end]:
        duration = item["duration"]
        length = duration * sample_rate
        if length < chunk_length:
            sample_index = -1
            if length * 2 < chunk_length and length * 4 > chunk_length:
                sample_index = -2
            elif length * 2 > chunk_length:
                sample_index = -1
            else:
                continue
            result_list.append([item, sample_index])
        else:
            sample_index = 0
            while sample_index + chunk_length <= length:
                result_list.append([item, sample_index])
                sample_index += chunk_length
            if sample_index < length:
                result_list.append([item, int(length - chunk_length)])


def collate_fn(batch):
    label_mel_specs = [item["label_mel_spec"].squeeze(0) for item in batch]
    label_mel_lengths = torch.LongTensor([spec.shape[-1] for spec in label_mel_specs])
    max_mel_length = label_mel_lengths.amax()

    padded_label_mel_specs = []
    for spec in label_mel_specs:  
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_label_mel_specs.append(padded_spec)

    label_mel_specs = torch.stack(padded_label_mel_specs)

    noisy_mel_specs = [item["noisy_mel_spec"].squeeze(0) for item in batch]
    noisy_mel_lengths = torch.LongTensor([spec.shape[-1] for spec in noisy_mel_specs])
    max_mel_length = noisy_mel_lengths.amax()

    padded_noisy_mel_specs = []
    for spec in noisy_mel_specs:  
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_noisy_mel_specs.append(padded_spec)

    noisy_mel_specs = torch.stack(padded_noisy_mel_specs)
    label_paths = [item['label_path'] for item in batch]

    return dict(
        label_mel=label_mel_specs,
        noisy_mel=noisy_mel_specs,
        label_mel_lengths=label_mel_lengths,
        noisy_mel_lengths=noisy_mel_lengths,
        label_paths=label_paths,
    )

        
class DynamicBatchSampler(Sampler[list[int]]):
    """
    基于总帧数的动态 Batch 划分机制：
    1. 动态调节 batch size，保证显存占用基于总帧数进行阈值控制。
    2. 控制同一个 batch 内样本长度差，提高 padding 效率。
    """
    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False, cache_path: str = None
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.max_diff = 1.5
        
        if cache_path is not None and os.path.exists(cache_path):
            self.batches = torch.load(cache_path)
            return
        
        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(self.sampler, desc="Sorting dataset..."):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1], reverse=True)

        batch = []
        batch_frames = 0
        
        for idx, frame_len in tqdm(indices, desc=f"Creating dynamic batches"):
            if len(batch) > 0:
                prev_frame_len = data_source.get_frame_len(batch[-1])  
                frame_diff = max(prev_frame_len/frame_len, frame_len/prev_frame_len) 
            else:
                frame_diff = float('inf')  
            
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples) and frame_diff <= self.max_diff:
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

        random.seed(random_seed)
        random.shuffle(batches)

        self.batches = batches

        if cache_path is not None:
            torch.save(self.batches, cache_path)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class DistributedBatchWrapper:
    """
    多卡分布式训练环境下的 Batch 索引分发器。
    支持断点续训时通过跳过已处理的 batch 实现快速对齐。
    """
    def __init__(self, batch_sampler, num_replicas, rank):
        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.batch_sampler) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.skip_batches = 0 

    def set_skip_batches(self, n):
        self.skip_batches = n

    def __iter__(self):
        batches = list(self.batch_sampler)
        g = torch.Generator()
        g.manual_seed(42 + self.epoch)
        indices = torch.randperm(len(batches), generator=g).tolist()
        
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size] 
            
        subsampled_indices = indices[self.rank :: self.num_replicas]
        
        if self.skip_batches > 0:
            subsampled_indices = subsampled_indices[self.skip_batches:]
            self.skip_batches = 0 
        
        for i in subsampled_indices:
            yield batches[i]

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


def make_auto_loader(
    clean_scp,
    regular_noise_scp,
    rir_scp,
    text_scp,
    repeat=1,
    batch_size=8,
    max_samples=64,
    num_workers=4,
    sample_rate=16000,
    probability=None,
    snr_ranges=None,
    scale_ranges=None,
    local_rank=None,
    world_size=None,
):
    if local_rank is None:
        local_rank = dist.get_rank() if dist.is_initialized() else 0
    if world_size is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1

    dataset = AutoDataset(
        clean_scp=clean_scp,
        text_scp=text_scp,
        regular_noise_scp=regular_noise_scp,
        rir_scp=rir_scp,
        repeat=repeat,
        num_workers=num_workers,
        sample_rate=sample_rate,
        probability=probability,
        snr_ranges=snr_ranges,
        scale_ranges=scale_ranges,
    )
    
    sampler = SequentialSampler(dataset)
    cache_path = f"{clean_scp}_bs{batch_size}_ms{max_samples}.batches.pt"

    batch_sampler = DynamicBatchSampler(
        sampler, batch_size, max_samples=max_samples, random_seed=42, drop_last=True, cache_path=cache_path
    )

    if world_size > 1:
        batch_sampler = DistributedBatchWrapper(batch_sampler, world_size, local_rank)

    def parallel_collate_fn(indices):
        """接管 collate 阶段的 IO 请求，通过线程池实现音频的并行预加载，这里的max_workers代表一个worker所调用的线程数"""
        with ThreadPoolExecutor(max_workers=16) as executor:
            samples = list(executor.map(dataset.__getitem__, indices))
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except:
            pass
        return collate_fn(samples)

    loader = DataLoader(
        IndexOnlyDataset(len(dataset)),
        collate_fn=parallel_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else None,
        batch_sampler=batch_sampler,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return batch_sampler, loader