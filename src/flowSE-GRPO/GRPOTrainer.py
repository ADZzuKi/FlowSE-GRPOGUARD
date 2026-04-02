import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from peft import LoraConfig, get_peft_model 

class GRPOTrainer:
    def __init__(
        self,
        base_model: nn.Module,
        vocoder: nn.Module,
        dnsmos_model: nn.Module, 
        lr: float = 2e-4,
        group_size: int = 12,    
        ppo_epochs: int = 2,     
        clip_epsilon: float = 0.2, 
        beta_kl: float = 0.05,   
        sde_a: float = 0.4,      
        max_steps: int = 5000,   
    ):
        self.device = next(base_model.parameters()).device
        
        for param in base_model.parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.0.0", "ff.2"],
            lora_dropout=0.0,
            bias="none",
            init_lora_weights="gaussian"
        )
        base_model.transformer = get_peft_model(base_model.transformer, lora_config)
        self.model = base_model 
        self.model.train() 
        
        self.vocoder = vocoder.eval()
        self.dnsmos_model = dnsmos_model.eval()
        
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = AdamW(trainable_params, lr=lr)

        def lr_lambda(current_step: int):
            return max(0.0, 1.0 - float(current_step) / float(max_steps))
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.G = group_size
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_epsilon
        self.beta_kl = beta_kl
        self.sde_a = sde_a

    @torch.no_grad()
    def get_rewards(self, mel_g, clean_audio_g):
        """Decode mel-spectrograms and evaluate using the DNSMOS reward model."""
        wavs = self.vocoder.decode(mel_g.permute(0, 2, 1).to(torch.float32)) 
        c_wavs = clean_audio_g.to(torch.float32)

        def norm(w):
            rms = torch.sqrt(torch.mean(w**2, dim=-1, keepdim=True))
            return w * (10**(-25/20) / (rms + 1e-8))
            
        wavs = norm(wavs)
        c_wavs = norm(c_wavs)
        
        rewards, logs = self.dnsmos_model(wavs, c_wavs) 
        return rewards, logs

    def compute_group_advantages(self, rewards, batch_size):
        """Compute relative advantages within each group for GRPO."""
        rewards_bg = rewards.view(batch_size, self.G)
        mean_r = rewards_bg.mean(dim=1, keepdim=True)
        std_r = rewards_bg.std(dim=1, keepdim=True) + 1e-8 
        advantages = (rewards_bg - mean_r) / std_r
        return advantages.view(-1).detach()

    def train_step(self, cond_batch: torch.Tensor, clean_audio: torch.Tensor, steps: int = 12):
        if torch.isnan(cond_batch).any() or torch.isnan(clean_audio).any():
            raise ValueError("NaN detected in input audio conditions.")

        B = cond_batch.shape[0]
        
        # -------- Phase 1: Rollout --------
        self.model.eval() 
        cond_g = cond_batch.repeat_interleave(self.G, dim=0)
        clean_audio_g = clean_audio.repeat_interleave(self.G, dim=0)
        
        with torch.no_grad():
            mel_g, _, rl_states = self.model.sample_rl(
                cond=cond_g, steps=steps, cfg_strength=0.0, 
                sde_kwargs=dict(noise_level_a=self.sde_a, sde_window_start=1, sde_window_size=2)
            )
        
        if torch.isnan(mel_g).any():
            raise ValueError("NaN detected during SDE generation.")

        num_demos = min(2, B)
        demo_indices = [i * self.G for i in range(num_demos)]
        demo_mels = mel_g[demo_indices].detach().clone()
        
        # -------- Phase 2: Reward Computation --------
        rewards, raw_metrics_log = self.get_rewards(mel_g, clean_audio_g)
        
        if torch.isnan(rewards).any():
            raise ValueError("NaN detected in computed rewards.")

        advantages = self.compute_group_advantages(rewards, B)
        
        metrics_log = {}
        adv_bg = advantages.view(B, self.G)
        metrics_log["Advantage_Batch/Mean_Abs"] = advantages.abs().mean().item()
        metrics_log["Advantage_Batch/Range_Mean"] = (adv_bg.max(dim=1)[0] - adv_bg.min(dim=1)[0]).mean().item()

        rew_bg = rewards.view(B, self.G)
        metrics_log["Reward_Combined_Batch/Mean"] = rewards.mean().item()
        metrics_log["Reward_Combined_Batch/Range_Mean"] = (rew_bg.max(dim=1)[0] - rew_bg.min(dim=1)[0]).mean().item()

        for metric_name, v_list in raw_metrics_log.items():
            vals_bg = torch.tensor(v_list, dtype=torch.float32).view(B, self.G)
            metrics_log[f"Raw_{metric_name}_Batch/Mean"] = vals_bg.mean().item()
            metrics_log[f"Raw_{metric_name}_Batch/Range_Mean"] = (vals_bg.max(dim=1)[0] - vals_bg.min(dim=1)[0]).mean().item()
        
        # -------- Phase 3: Policy Optimization --------
        self.model.train() 
        
        log_prob_old_list = []
        with torch.no_grad():
            for state in rl_states:
                lp_old, _, _ = self.model.forward_rl(
                    x_t=state['x_t'], cond=cond_g, t=state['t_tensor'], 
                    x_next=state['x_next'], dt=state['dt'], a=self.sde_a
                )
                log_prob_old_list.append(lp_old.detach())

        total_policy_loss = 0.0
        total_kl_div = 0.0

        for epoch in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            
            for step_idx, state in enumerate(rl_states):
                curr_x_t = state['x_t'].detach()
                curr_x_t.requires_grad_(True)
                dt_val = state['dt']
                
                log_prob_theta, x_mean_theta, std_dev_t = self.model.forward_rl(
                    x_t=curr_x_t, cond=cond_g, t=state['t_tensor'], 
                    x_next=state['x_next'], dt=dt_val, a=self.sde_a
                )

                with self.model.transformer.disable_adapter():
                    with torch.no_grad(): 
                        _, x_mean_ref, _ = self.model.forward_rl(
                            x_t=state['x_t'], cond=cond_g, t=state['t_tensor'], 
                            x_next=state['x_next'], dt=dt_val, a=self.sde_a
                        )

                # RatioNorm bias correction to center ratio mean at 1.0
                ratio_mean_bias = (x_mean_theta - state['x_mean']).pow(2).mean(dim=(1, 2))
                ratio_mean_bias = ratio_mean_bias / (2 * std_dev_t.mean() ** 2)
                
                ratio = torch.exp((log_prob_theta - log_prob_old_list[step_idx] + ratio_mean_bias) * std_dev_t.mean())

                # Clipped PPO loss computation
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                
                policy_loss = policy_loss / dt_val

                # KL divergence estimation via MSE between equal-variance Gaussian means
                kl_loss = ((x_mean_theta - x_mean_ref) ** 2).mean(dim=(1, 2)) / (2 * std_dev_t.mean() ** 2)
                kl_loss = torch.mean(kl_loss)

                loss = policy_loss + self.beta_kl * kl_loss
                loss.backward()

                total_policy_loss += policy_loss.item()
                total_kl_div += kl_loss.item()
                
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.scheduler.step()
        num_updates = self.ppo_epochs * len(rl_states)
        metrics_log["Loss/Policy"] = total_policy_loss / num_updates
        metrics_log["Loss/KL_Div"] = total_kl_div / num_updates
        metrics_log["Optim/Learning_Rate"] = self.optimizer.param_groups[0]['lr'] 
        
        return metrics_log, demo_mels