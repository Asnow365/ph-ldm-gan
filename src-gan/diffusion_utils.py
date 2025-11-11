# diffusion_utils.py

import torch

# ————————————————
# 1) get_time_schedule
# 给定总步数 n_timesteps，返回 alpha_bars（累积乘积）和 betas
# 这个实现直接参考 LatentDiffusion.py 中对应的逻辑。
def get_time_schedule(n_timesteps, device):
    import numpy as np
    eps_small = 1e-3
    t = torch.from_numpy(np.linspace(eps_small, 1.0, n_timesteps + 1, dtype=np.float64)).to(device)
    # 这里假设 beta_min=0.1, beta_max=20.0（你可以按需改）
    beta_min, beta_max = 0.1, 20.0
    var = t * (beta_max - beta_min) + beta_min   # 线性调度（示例），或者用几何/VP
    alpha_bars = 1.0 - var
    return alpha_bars, var  # var 这里暂且当 betas 用

# ————————————————
# 2) get_posterior_coeff
# 给定 alpha_cumprod（我们这里直接把 alpha_bars 传进来），
# 计算后验所需的 coef1, coef2, var。逻辑参考 LatentDiffusion.py。
def get_posterior_coeff(alpha_bars: torch.Tensor):
    alphas = alpha_bars[1:]  # 忽略 a_0=1
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=alphas.device), alphas_cumprod[:-1]], dim=0)
    posterior_variance = alphas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    posterior_mean_coef1 = alphas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
    posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)
    posterior_log_var_clipped = torch.log(posterior_variance.clamp(min=1e-20))
    return posterior_mean_coef1, posterior_mean_coef2, posterior_variance

# ————————————————
# 3) q_sample_pairs
# 给定一个“forward_process”对象，或者直接给 alpha_bars/sigmas/betas，
# 对 x0 做一次前向噪声扩散，返回 (x_t, x_{t+1})。
# 这里示例用最简单的公式：x_t = sqrt(alpha_cumprod[t]) * x0 + sqrt(1-alpha_cumprod[t]) * noise。
def q_sample_pairs(forward_process, x_start: torch.Tensor, t: torch.Tensor):
    # 如果 forward_process 里已经封装好 a_s, sigmas，就用下面这两行：
    # x_t = forward_process.q_sample(x_start, t)
    # x_t_plus_one = forward_process.q_sample(x_start, t + 1)
    # 但为了最简，我们假设 forward_process 直接是 (alpha_bars, betas)。
    alpha_bars, betas = forward_process
    # alpha_bars[t] 对应第 t 步的累积 alpha
    a_t   = alpha_bars[t]                           # (B,)
    a_t1  = alpha_bars[t + 1]                       # (B,)
    noise = torch.randn_like(x_start)
    a_t   = a_t.view(-1, 1, 1, 1)
    a_t1  = a_t1.view(-1, 1, 1, 1)
    # x_t
    x_t = torch.sqrt(a_t) * x_start + torch.sqrt(1 - a_t) * noise
    # x_{t+1}
    x_tp1 = torch.sqrt(a_t1) * x_start + torch.sqrt(1 - a_t1) * noise
    return x_t, x_tp1

# ————————————————
# 4) sample_posterior
# 给定 posterior_coeff（coef1, coef2, var）和 x0_pred, x_t, t，
# 计算一次后验采样 p(x_t | x_{t+1})
def sample_posterior(posterior_coeff, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
    coef1 = posterior_coeff.posterior_mean_coef1
    coef2 = posterior_coeff.posterior_mean_coef2
    var   = posterior_coeff.posterior_variance
    # 先取出对应 t 位置的比例系数
    m1 = coef1[t].view(-1, 1, 1, 1)
    m2 = coef2[t].view(-1, 1, 1, 1)
    v  = var[t].view(-1, 1, 1, 1)
    mean = m1 * x_0 + m2 * x_t
    noise = torch.randn_like(x_t)
    nonzero = (t != 0).float().view(-1, 1, 1, 1)
    return mean + nonzero * torch.sqrt(v) * noise
