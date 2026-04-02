import torch

class FlowSESampler:
    """
    Hybrid ODE-SDE sampler implemented for FLOWSE-GRPO.
    """
    def __init__(self, steps=10, noise_level_a=0.4, sde_window_start=1, sde_window_size=2):
        self.steps = steps
        self.a = noise_level_a
        self.sde_window_start = sde_window_start
        self.sde_window_size = sde_window_size

    @torch.no_grad()
    def sample(self, model_fn, y0):
        device = y0.device
        x = y0
        dt = 1.0 / self.steps
        
        trajectory = [x]
        rl_states = [] 
        
        for i in range(self.steps):
            t_val = i / self.steps
            t_tensor = torch.full((1,), t_val, device=device, dtype=x.dtype)
            
            v_theta = model_fn(t_tensor, x)
            in_sde_window = (self.sde_window_start <= i < self.sde_window_start + self.sde_window_size)
            
            if in_sde_window and t_val > 0:
                # SDE Update: sigma_t = a * sqrt((1 - t) / t)
                sigma_t = self.a * torch.sqrt(torch.tensor((1.0 - t_val) / t_val, device=device))
                
                # Drift correction: [sigma_t^2 / 2(1-t)] * (-x + t * v_theta)
                drift_correction = (sigma_t ** 2) / (2 * (1.0 - t_val)) * (-x + t_val * v_theta)
                x_mean = x + (v_theta + drift_correction) * dt
                
                # Brownian motion injection
                epsilon = torch.randn_like(x)
                std_dev = sigma_t * torch.sqrt(torch.tensor(dt, device=device))
                x_next = x_mean + std_dev * epsilon
                
                # Save context for GRPO Actor network log_prob re-computation
                rl_states.append({
                    'step_idx': i,
                    't_tensor': t_tensor,
                    'x_t': x.detach().clone(),
                    'v_theta_detach': v_theta.detach().clone(), 
                    'x_mean': x_mean.detach().clone(), 
                    'x_next': x_next.detach().clone(),
                    'std_dev': std_dev,
                    'dt': dt
                })
                
            else:
                # Standard Euler ODE Update
                x_next = x + v_theta * dt
                
            x = x_next
            trajectory.append(x)
            
        return x, trajectory, rl_states