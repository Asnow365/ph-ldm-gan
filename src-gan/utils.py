# utils.py

import torch
import torch.nn.functional as F
import numpy as np

# ==============================
# —— 1. Variance Schedule (beta) 工具
# ==============================

def var_func_vp(t, beta_min, beta_max):
    """
    VP（Variance Preserving）调度：var(t) = beta_min + t*(beta_max - beta_min)
    t: 1D tensor in [0,1]
    返回同样 shape 的 var(t)。
    """
    return beta_min + t * (beta_max - beta_min)

def var_func_geometric(t, beta_min, beta_max):
    """
    几何调度：var(t) = beta_min * (beta_max / beta_min) ^ t
    t: 1D tensor in [0,1]
    """
    return beta_min * ((beta_max / beta_min) ** t)

def extract(arr, t, shape):
    """
    从一维张量 arr 中按照索引 t 抽出相应元素，然后 reshape 到 `shape`，以便广播。
    - arr: 一维 tensor，比如长度 (T+1,)
    - t:   长度为 batch_size 的整型 tensor，表示每个样本使用哪一个索引
    - shape: 目标形状，比如 (B, C, H, W)，最终 reshape 成 (B,1,1,1,...)
    返回：一个 tensor，可以直接用于与形状 (B,C,H,W) 的数据做逐元素乘法。
    """
    batch_size = t.shape[0]
    out = arr.gather(0, t)  # shape = (batch_size,)
    return out.view(batch_size, *([1] * (len(shape) - 1)))

def get_time_schedule(n_timesteps, device, beta_min=0.1, beta_max=20.0, use_geometric=False):
    """
    计算扩散过程中的 alpha_bars 和 betas：
      - n_timesteps: 总步数，例如 1000
      - beta_min, beta_max: beta 的最小/最大值
      - use_geometric: 如果 True 则使用几何调度，否则使用 VP 线性调度
    返回：
      alpha_bars: 长度 (n_timesteps+1) 的 tensor，alpha_bars[t] = ∏_{i=1}^t (1 - beta_i)
      betas:      长度 (n_timesteps+1) 的 tensor，其中 betas[0] = 1e-8，后续按 alpha_bars 反推：
                   betas[t] = 1 - alpha_bars[t] / alpha_bars[t-1]
    """
    eps_small = 1e-3
    # 先构造从 eps_small 到 1.0 的 t 刻度
    t = torch.linspace(eps_small, 1.0, steps=n_timesteps + 1, device=device, dtype=torch.float64)
    if use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var  # alpha_bars[t] = ∏ alpha_i（等价于 1 - var）

    betas = torch.zeros(n_timesteps + 1, device=device, dtype=torch.float32)
    betas[0] = 1e-8
    # betas[t] = 1 - alpha_bars[t] / alpha_bars[t-1]
    for i in range(1, n_timesteps + 1):
        betas[i] = 1.0 - float(alpha_bars[i] / alpha_bars[i - 1])
    betas = betas.clamp(min=1e-8)
    return alpha_bars.float().to(device), betas

def get_posterior_coeff(alpha_bars):
    """
    给定 alpha_bars（长度 T+1），计算后验采样所需的系数：
      - posterior_mean_coef1[t] = beta_t * sqrt(alpha_bar[t-1]) / (1 - alpha_bar[t])
      - posterior_mean_coef2[t] = (1 - alpha_bar[t-1]) * sqrt(alpha_t) / (1 - alpha_bar[t])
      - posterior_variance[t]  = beta_t * (1 - alpha_bar[t-1]) / (1 - alpha_bar[t])
    其中 alpha_t = alpha_bar[t] / alpha_bar[t-1]，beta_t = 1 - alpha_t.

    返回：
      posterior_mean_coef1, posterior_mean_coef2, posterior_variance —— 都是 shape=(T+1,) 的 tensor。
      （第 0 项是占位，不会实际使用，都是 0 或 1e-8。）
    """
    device = alpha_bars.device
    betas = 1.0 - alpha_bars  # 这样 α_t = alpha_bars[t] / alpha_bars[t-1]
    # 真实的 alphas、alpha_cumprod
    alphas = alpha_bars[1:]
    alphas_cumprod = alpha_bars[1:].cumprod(dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), alphas_cumprod[:-1]], dim=0)

    posterior_variance = betas[1:] * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    posterior_mean_coef1 = betas[1:] * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
    posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)

    # 补齐到长度 T+1，index 0 不用
    posterior_variance = torch.cat([torch.tensor([1e-8], device=device), posterior_variance], dim=0)
    posterior_mean_coef1 = torch.cat([torch.tensor([0.], device=device), posterior_mean_coef1], dim=0)
    posterior_mean_coef2 = torch.cat([torch.tensor([0.], device=device), posterior_mean_coef2], dim=0)
    return posterior_mean_coef1, posterior_mean_coef2, posterior_variance

# ==============================
# —— 2. 前向扩散采样函数 q_sample_pairs
# ==============================

def q_sample_pairs(coeff, x_start, t):
    """
    给定：
      coeff: 一个包含属性 a_s (length T+1) 和 sigmas (length T+1) 的对象，
             其中 a_s[t] = sqrt(1 - beta_t)，sigmas[t] = sqrt(beta_t)
      x_start: clean image x0, shape = (B, C, H, W)
      t:  当前时间步索引，tensor shape = (B,) int
    返回：
      x_t, x_{t+1} 两对张量，shape 都是 (B, C, H, W)
    计算：
      x_t  = a_s[t]  * x0 + sigma[t]  * noise
      x_t1 = a_s[t+1]* x0 + sigma[t+1] * noise   （使用同一 noise）
    """
    a_s = coeff.a_s  if hasattr(coeff, 'a_s') else coeff[0]
    sigmas = coeff.sigmas if hasattr(coeff, 'sigmas') else coeff[1]
    noise = torch.randn_like(x_start)
    a_t   = extract(a_s,   t, x_start.shape)
    sigma_t  = extract(sigmas, t, x_start.shape)
    a_t1  = extract(a_s,   t + 1, x_start.shape)
    sigma_t1 = extract(sigmas, t + 1, x_start.shape)
    x_t  = a_t   * x_start + sigma_t   * noise
    x_t1 = a_t1  * x_start + sigma_t1  * noise
    return x_t, x_t1

# ==============================
# —— 3. 后验采样函数 sample_posterior
# ==============================

def sample_posterior(posterior_coeff, x_0, x_t, t):
    """
    给定：
      posterior_coeff: 一个包含属性 posterior_mean_coef1 (shape T+1),
                                         posterior_mean_coef2 (shape T+1),
                                         posterior_variance      (shape T+1)
      x_0:   生成器预测的 clean latent x0_hat, shape = (B,C,H,W)
      x_t:   当前 noisy latent, shape = (B,C,H,W)
      t:     当前时间步索引 tensor shape = (B,)
    计算一次 p(x_{t-1} | x_t, x0_hat) 的采样：
      mean = coef1[t] * x0_hat + coef2[t] * x_t
      var  = posterior_variance[t]
      x_{t-1} = mean + sqrt(var) * noise  （对于 t=0 强制输出 x0，没有 noise）
    返回：
      x_{t-1} tensor, shape = (B, C, H, W)
    """
    coef1 = posterior_coeff.posterior_mean_coef1
    coef2 = posterior_coeff.posterior_mean_coef2
    var   = posterior_coeff.posterior_variance
    m1 = extract(coef1, t, x_t.shape)
    m2 = extract(coef2, t, x_t.shape)
    v  = extract(var,   t, x_t.shape)
    mean = m1 * x_0 + m2 * x_t
    noise = torch.randn_like(x_t)
    nonzero = (t != 0).float().view(-1, 1, 1, 1)
    return mean + nonzero * torch.sqrt(v) * noise

# ==============================
# —— 4. GAN Losses & 正则化（R1, Relativistic, 常规 GAN Loss）
# ==============================

def r1_penalty(real_outputs, real_inputs):
    """
    计算 R1 零中心梯度惩罚：
      real_outputs: D(real_inputs) 的输出 logits，shape = (B,)
      real_inputs:  需要对它求梯度的输入，shape = (B, C, H, W)
    返回：
      penalty = 0.5 * E[ || ∇_{real_inputs} D(real_inputs) ||^2 ]
    """
    grads = torch.autograd.grad(
        outputs=real_outputs.sum(), inputs=real_inputs, create_graph=True
    )[0]
    grad_pen = (grads.view(grads.shape[0], -1).norm(2, dim=1) ** 2).mean() * 0.5
    return grad_pen

def relativistic_loss_D(real_logits, fake_logits):
    """
    Relativistic Discriminator Loss（Softplus 版）：
      L_D = E[Softplus(-(real_logits - fake_logits))]
    real_logits, fake_logits: shape = (B,)
    """
    rel = real_logits - fake_logits
    return F.softplus(-rel).mean()

def relativistic_loss_G(real_logits, fake_logits):
    """
    Relativistic Generator Loss：
      L_G = E[Softplus(fake_logits - real_logits)]
    """
    rel = fake_logits - real_logits
    return F.softplus(-rel).mean()

def gan_loss(predictions, is_real):
    """
    标准 Softplus GAN Loss：
      如果 is_real=True,  L = Softplus(-predictions).mean()
      否则           L = Softplus(predictions).mean()
    """
    if is_real:
        return F.softplus(-predictions).mean()
    else:
        return F.softplus(predictions).mean()

# ============ END of utils.py ===========
