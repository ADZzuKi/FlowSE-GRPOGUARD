import os
import sys
import argparse
import time
import yaml
import pprint
import random
import glob
import re
import gc
import ctypes
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.optim import AdamW

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from vocos import Vocos
from utils.logger import get_logger
from model import DiT, CFM
from loader.dataloader import make_auto_loader

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        base_model = model.module if hasattr(model, 'module') else model
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
                
    def update(self, model):
        base_model = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if param.requires_grad:
                    # Exponential Moving Average update
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
                    
    def state_dict(self):
        return self.shadow
        
    def load_state_dict(self, state_dict):
        for name in self.shadow:
            if name in state_dict:
                self.shadow[name].copy_(state_dict[name])


def make_dataloader(opt, local_rank, world_size):
    train_sampler, train_loader = make_auto_loader(
        **opt["datasets"]["train"],
        **opt["datasets"]["dataloader_setting"],
        local_rank=local_rank,
        world_size=world_size
    )
    
    val_sampler, val_loader = make_auto_loader(
        **opt["datasets"]["val"],
        **opt["datasets"]["dataloader_setting"],
        local_rank=local_rank,
        world_size=world_size
    )
    return train_sampler, train_loader, val_sampler, val_loader


def save_checkpoint(checkpoint_dir, nnet, optimizer, scheduler, epoch, best_loss, step=None, save_period=-1, best=True, logger=None, ema=None):
    cpt = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": nnet.module.state_dict() if hasattr(nnet, 'module') else nnet.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),  
        "best_loss": best_loss,
    }
    
    if ema is not None:
        cpt["ema_state_dict"] = ema.state_dict()
        
    if step is not None:
        torch.save(cpt, checkpoint_dir / f"step_{step}.pt.tar")
        if logger: logger.info(f"Saved checkpoint: step_{step}.pt.tar")
    else:
        cpt_name = "{0}.pt.tar".format("best" if best else "last")
        torch.save(cpt, checkpoint_dir / cpt_name)
        if logger: logger.info(f"Saved epoch checkpoint: {cpt_name}")


def load_obj(obj, device):
    def cuda(obj):
        return obj.to(device, non_blocking=True) if isinstance(obj, torch.Tensor) else obj
    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def plot_spectrogram(mel_tensor):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mel_tensor.cpu().numpy(), aspect="auto", origin="lower", interpolation='none')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return np.transpose(data, (2, 0, 1))


@torch.no_grad()
def run_visual_validation(nnet, val_loader, writer, step, device, vocoder, logger):
    nnet.eval()
    batch = next(iter(val_loader))
    
    num_samples = min(4, batch['noisy_mel'].size(0))
    noisy_mel = batch['noisy_mel'][:num_samples].to(device)
    clean_mel = batch['label_mel'][:num_samples].to(device)
    noisy_lens = batch['noisy_mel_lengths'][:num_samples].to(device)
    
    noisy_input = noisy_mel.transpose(-1, -2) 
    
    try:
        base_model = nnet.module if hasattr(nnet, 'module') else nnet
        if hasattr(base_model, 'sample'):
            sample_out = base_model.sample(noisy_input)
            enhanced_mel = sample_out[0].transpose(-1, -2) if isinstance(sample_out, tuple) else sample_out.transpose(-1, -2)
        else:
            nnet.train()
            return
    except Exception as e:
        logger.error(f"Inference sampling failed: {e}")
        nnet.train()
        return

    for idx in range(num_samples):
        valid_len = noisy_lens[idx]
        gen_mel = enhanced_mel[idx:idx+1, :, :valid_len]
        ref_mel = clean_mel[idx:idx+1, :, :valid_len]
        nsy_mel = noisy_mel[idx:idx+1, :, :valid_len]
        
        gen_wav = vocoder.decode(gen_mel).squeeze().cpu().unsqueeze(0)
        ref_wav = vocoder.decode(ref_mel).squeeze().cpu().unsqueeze(0)
        nsy_wav = vocoder.decode(nsy_mel).squeeze().cpu().unsqueeze(0)
        
        writer.add_audio(f"Audio_Step_{step}/1_Noisy_{idx}", nsy_wav, step, sample_rate=24000)
        writer.add_audio(f"Audio_Step_{step}/2_Enhanced_{idx}", gen_wav, step, sample_rate=24000)
        writer.add_audio(f"Audio_Step_{step}/3_Clean_{idx}", ref_wav, step, sample_rate=24000)
        
        writer.add_image(f"Spec_Step_{step}/1_Noisy_{idx}", plot_spectrogram(nsy_mel[0]), step)
        writer.add_image(f"Spec_Step_{step}/2_Enhanced_{idx}", plot_spectrogram(gen_mel[0]), step)
        writer.add_image(f"Spec_Step_{step}/3_Clean_{idx}", plot_spectrogram(ref_mel[0]), step)

    nnet.train()


def validate_one_epoch(val_loader, nnet, local_rank, conf, device, world_size, logger):
    losses = AverageMeter("Loss", ":.4f")
    nnet.eval()
    base_model = nnet.module if hasattr(nnet, 'module') else nnet
    
    gc.collect()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for i, egs in enumerate(val_loader):
            if i >= 20: 
                break
                
            egs = load_obj(egs, device)
            noisy = egs["noisy_mel"].transpose(-1, -2)
            label = egs["label_mel"].transpose(-1, -2)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _, _ = base_model(inp=noisy, clean=label)
            
            losses.update(loss.item(), noisy.size(0))
            
    return losses.avg


def train_one_epoch(
    train_loader, val_loader, nnet, optimizer, scheduler, epoch, 
    local_rank, conf, device, world_size, logger, global_step, 
    writer, vocoder, checkpoint_dir, training_start_time, ema=None
):
    nnet.train()
    
    from tqdm import tqdm
    if local_rank == 0:
        pbar = tqdm(train_loader, initial=(global_step % len(train_loader)), desc=f"global_step {global_step}", mininterval=5.0, dynamic_ncols=True)
        iterator = enumerate(pbar)
    else:
        pbar = None
        iterator = enumerate(train_loader)

    for i, egs in iterator:
        egs = load_obj(egs, device)
        noisy = egs["noisy_mel"].transpose(-1, -2)
        label = egs["label_mel"].transpose(-1, -2)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, _, _ = nnet(inp=noisy, clean=label)

        reduced_loss = reduce_mean(loss, world_size)
        
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(nnet.parameters(), conf["optim"]["gradient_clip"])
        
        optimizer.step()
        scheduler.step()
        
        if ema is not None:
            ema.update(nnet)

        global_step += 1
        save_steps = conf["train"].get("save_checkpoint_steps", 5000)
        
        if global_step % save_steps == 0:
            if local_rank == 0: 
                original_weights = {}
                base_model = nnet.module if hasattr(nnet, 'module') else nnet
                
                # Apply EMA weights temporarily for validation stability
                if ema is not None:
                    with torch.no_grad():
                        for name, param in base_model.named_parameters():
                            if param.requires_grad:
                                original_weights[name] = param.data.clone()
                                if name in ema.shadow:
                                    param.data.copy_(ema.shadow[name])

                cv_loss = validate_one_epoch(val_loader, nnet, local_rank, conf, device, world_size, logger)
                save_checkpoint(checkpoint_dir, nnet, optimizer, scheduler, epoch, best_loss=0, step=global_step, best=False, logger=logger, ema=ema)
                run_visual_validation(nnet, val_loader, writer, global_step, device, vocoder, logger)
                
                writer.add_scalar("Loss/Val", cv_loss, global_step)

                # Restore original model weights
                if ema is not None:
                    with torch.no_grad():
                        for name, param in base_model.named_parameters():
                            if param.requires_grad and name in original_weights:
                                param.data.copy_(original_weights[name])
            
            nnet.train()
            dist.barrier()

        if local_rank == 0:
            if pbar is not None:
                pbar.set_description(f"Step {global_step}")
                pbar.set_postfix({
                    'Loss': f"{reduced_loss.item():.4f}", 
                    'LR': f"{optimizer.param_groups[0]['lr']:.2e}",
                    "running_time": int(time.monotonic() - training_start_time)
                })

            if global_step % 50 == 0:
                writer.add_scalar("Loss/Train", reduced_loss.item(), global_step)
                writer.add_scalar("LR/Train", get_learning_rate(optimizer), global_step)
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except:
                    pass
        
        # DDP Timeout Sync Mechanism
        timeout_flag = torch.tensor([0], dtype=torch.int32, device=device)
        
        if local_rank == 0:
            current_duration = time.monotonic() - training_start_time
            if current_duration > 84600: 
                timeout_flag[0] = 1
                logger.info(f"Timeout threshold reached ({current_duration/3600:.2f}h). Initiating graceful shutdown.")
        
        dist.broadcast(timeout_flag, src=0)
        
        if timeout_flag.item() == 1:
            if local_rank == 0:
                save_checkpoint(checkpoint_dir, nnet, optimizer, scheduler, epoch, best_loss=0, step=global_step, best=False, logger=logger, ema=ema)
            dist.barrier()
            sys.exit(0)
            
    return global_step


def main_worker(local_rank, args):
    dist.init_process_group(backend="nccl")
    training_start_time = time.monotonic()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cudnn.benchmark = True
    world_size = dist.get_world_size()

    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    random.seed(conf['train']['seed'])
    np.random.seed(conf['train']['seed'])
    torch.cuda.manual_seed_all(conf['train']['seed'])
    
    checkpoint_dir = Path(conf["train"]["checkpoint"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    logger = get_logger(name=(checkpoint_dir / "trainer.log").as_posix(), file=True)
    
    if local_rank == 0:
        tb_dir = checkpoint_dir / "tensorboard_logs"
        tb_dir.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=tb_dir.as_posix())
        
        vocoder_conf = conf['model']['vocoder']
        if vocoder_conf.get('is_local', False):
            local_vocos_dir = vocoder_conf['local_path']
            vocoder = Vocos.from_hparams(f"{local_vocos_dir}/config.yaml")
            state_dict = torch.load(f"{local_vocos_dir}/pytorch_model.bin", map_location="cpu")
            vocoder.load_state_dict(state_dict)
        else:
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        vocoder = vocoder.to(device)
    else:
        writer, vocoder = None, None

    if local_rank == 0:
        logger.info("Arguments in yaml:\n{}".format(pprint.pformat(conf)))

    nnet = CFM(
        transformer=DiT(**conf['model']['arch'], mel_dim=conf['model']['mel_spec']['n_mel_channels']),
        audio_drop_prob=conf['model']['audio_drop_prob'], cond_drop_prob=conf['model']['cond_drop_prob'],
        mel_spec_kwargs=conf['model']['mel_spec']
    )
    
    start_epoch, global_step = 0, 0
    end_epoch = conf["train"]["epoch"]

    latest_ckpt = None
    ckpt_list = glob.glob(f"{checkpoint_dir}/step_*.pt.tar")
    if ckpt_list:
        ckpt_list.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)), reverse=True)
        latest_ckpt = ckpt_list[0]

    is_auto_resume = False
    resume_path = None

    if latest_ckpt:
        if local_rank == 0: logger.info(f"Auto-resuming from: {latest_ckpt}")
        resume_path = latest_ckpt
        is_auto_resume = True
    elif conf["train"]["resume"]:
        if local_rank == 0: logger.info(f"Loading base weights: {conf['train']['resume']}")
        resume_path = conf["train"]["resume"]
        is_auto_resume = False

    cpt = None
    global_step = 0  
    start_epoch = 0  
    
    if resume_path:
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        
        cpt = torch.load(resume_path, map_location="cpu")
        _saved_step = cpt.get("step", cpt.get("global_step", 0))
        
        pretrained_dict = cpt["model_state_dict"]
        model_dict = nnet.state_dict()
        
        if not is_auto_resume:
            # Truncate text projection layer dimension from 712 to 200 for base model compatibility
            filtered_dict = {k: v for k, v in pretrained_dict.items() if 'text' not in k}
            in_embed_weight_key = 'transformer.input_embed.proj.weight'
            if 'module.' + in_embed_weight_key in filtered_dict:
                in_embed_weight_key = 'module.' + in_embed_weight_key

            if in_embed_weight_key in filtered_dict:
                old_weight = filtered_dict[in_embed_weight_key]
                if old_weight.shape[1] > 200:
                    filtered_dict[in_embed_weight_key] = old_weight[:, :200]
            model_dict.update(filtered_dict)
        else:
            model_dict.update(pretrained_dict)
            global_step = _saved_step

        nnet.load_state_dict(model_dict, strict=False)
        
    nnet = nnet.cuda()
    ema = EMA(nnet, decay=0.9999)
    
    if resume_path and "ema_state_dict" in cpt:
        ema.load_state_dict(cpt["ema_state_dict"])
        
    nnet = DistributedDataParallel(nnet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = AdamW(nnet.parameters(), lr=conf['optim']['lr'])
    
    train_sampler, train_loader, val_sampler, val_loader = make_dataloader(conf, local_rank, world_size)

    if is_auto_resume and cpt is not None:
        start_epoch = global_step // len(train_loader)
        skip_steps = global_step % len(train_loader)
        
        if skip_steps > 0:
            train_loader.batch_sampler.set_skip_batches(skip_steps)
                
    warmup_steps = conf['optim']['warm_up_step'] * world_size
    total_steps = len(train_loader) * conf['optim']['max_epoch'] / conf['optim']['grad_accumulation_steps']
    decay_steps = total_steps - warmup_steps
    
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])
    
    if is_auto_resume and cpt is not None:
        optimizer.load_state_dict(cpt["optim_state_dict"])
        scheduler.load_state_dict(cpt["scheduler_state_dict"])  

    for epoch in range(start_epoch, end_epoch):
        if local_rank == 0: 
            logger.info(f"Begin train epoch: {epoch}")
        
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
            
        global_step = train_one_epoch(
            train_loader, val_loader, nnet, optimizer, scheduler, epoch, 
            local_rank, conf, device, world_size, logger, 
            global_step, writer, vocoder, checkpoint_dir, training_start_time, ema=ema
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowSE Distributed Training")
    parser.add_argument("-conf", type=str, required=True, help="YAML config file path")
    args = parser.parse_args()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    main_worker(local_rank, args)