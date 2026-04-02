# FlowSE & FlowSE-GRPO: Conditional Flow Matching for Speech Enhancement
This project is a personal implementation of the FlowSE-GRPO work.

Main references: 
FlowSE: https://arxiv.org/abs/2505.19476; 
FlowSE-GRPO: https://arxiv.org/abs/2601.16483; 
FlowGRPO: https://arxiv.org/abs/2505.05470; 
GRPO-GUARD: https://arxiv.org/abs/2510.22319

## Environment

'''bash
pip install -r requirement.txt

WeSpeaker should be installed via git+https://github.com/wenet-e2e/wespeaker.git

## Weights

### base model and vocoder
https://github.com/Honee-W/FlowSE

### SFT & Lora ckpt
https://huggingface.co/adzzuki/FlowSE-GRPOGUARD 

### Other weights you need to prepare
vocos: https://huggingface.co/charactr/vocos-mel-24khz
DNSMOS: https://github.com/microsoft/DNS-Challenge

## SFT Fine-tuning Stage
This stage fine-tunes the open-source FlowSE base model using your own high-quality dataset.

### Evaluation
'''bash
python noreverb_dnsmos_spk_wer.py \
    --config config/my_finetune.yaml \
    --ckpt_dir ./logs/exp_sft_flowSE \
    --dns3_dir ./datasets/test_set \
    --onnx_dir ./DNSMOS/ \
    --gt_feature_path ./datasets/test_set/clean_gt_features.pt

### SFT Training
'''bash
torchrun --nproc_per_node=1 train.py -conf config/my_finetune.yaml

Configurations to modify in config/my_finetune.yaml:
train.checkpoint: Directory to save training logs, TensorBoard and model weights.
train.resume: Path to the base pre-trained weights. The script will automatically truncate the text feature layer.
datasets.train.clean_scp: SCP file of clean speech data.
datasets.train.regular_noise_scp: SCP file of noise data.
model.vocoder.local_path: Local path to Vocos vocoder weights.

## FlowSE-GRPO
This stage is based on the SFT fine-tuned model, freezes the backbone network, injects LoRA as the policy model, and performs GRPO reinforcement learning alignment using DNSMOS-OVRL as the reward.

### Evaluation
The evaluation script automatically loads the SFT base model and dynamically hot-swaps the trained LoRA weights for decoding testing.

'''bash
python noreverb_metrics_eval.py \
    --config config/my_grpo.yaml \
    --lora_dir ./logs/exp_grpo_flowSE \
    --base_ckpt_path ./logs/exp_sft_flowSE/step_66000.pt.tar \
    --dns3_dir ./datasets/test_set \
    --onnx_dir ./DNSMOS/ \
    --gt_feature_path ./datasets/test_set/clean_gt_features.pt

### GRPO Training
'''bash
torchrun --nproc_per_node=1 train_grpo.py -conf config/my_grpo.yaml
Key configurations and path modifications:
Verify the following in config/my_grpo.yaml and train_grpo.py:
Base model source (train.resume): Must point to the best checkpoint from the SFT fine-tuning stage.