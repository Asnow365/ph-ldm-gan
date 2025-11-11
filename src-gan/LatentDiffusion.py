# -----------------------------------------------------------------------------
# LatentDiffusion_conditional_debug.py
# -----------------------------------------------------------------------------
# ⚠️ 这是在你刚刚提供的 **LatentDiffusion (1).py** 基础上做的 _最小增量修改版_
#    - 每一处改动都用  # <<< MOD  标记
#    - 主要目标：
#        1. 让调试信息 **更丰富**（梯度范数 / 参数量 / 学习率等）
#        2. 保证逻辑与原实现一致，绝不额外改变损失形式
# -----------------------------------------------------------------------------

import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T  #   <- 避免删除，后续可能用             # <<< MOD
from torch.utils.data import DataLoader

from diffusers.models import AutoencoderKL

from .DenoisingDiffusionProcess import (
    DenoisingDiffusionConditionalProcess,
)
from .networks import NLayerDiscriminator

# ───────────────────────────────────────────────
# Utility helpers for verbose debugging
# ───────────────────────────────────────────────

def _stat(name: str, tensor: torch.Tensor):
    if tensor is None:
        print(f"{name:>20s} | <None>")
        return
    print(
        f"{name:>20s} | shape={tuple(tensor.shape)} "
        f"mean={tensor.mean().item():.4f}  std={tensor.std().item():.4f} "
        f"min={tensor.min().item():.4f}  max={tensor.max().item():.4f}"
    )

def _grad_norm(module: nn.Module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.pow(2).sum().item()
    return math.sqrt(total) if total != 0 else None


class AutoEncoder(nn.Module):
    """Pre‑trained VAE wrapper (Stable‑Diffusion VAE)."""

    def __init__(self, model_type: str = "stabilityai/sd-vae-ft-ema"):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_type)

    def forward(self, x):
        return self.model(x).sample

    def encode(self, x, mode: bool = False):
        dist = self.model.encode(x).latent_dist
        return dist.mode() if mode else dist.sample()

    def decode(self, z):
        return self.model.decode(z).sample

class LatentDiffusionConditional(pl.LightningModule):
    def __init__(
        self,
        train_dataset,
        valid_dataset=None,
        num_timesteps: int = 1000,
        latent_scale_factor: float = 1.0,
        batch_size: int = 4,
        lr: float = 1e-4,
        gamma_r1: float = 1.0,
        debug: bool = True,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["train_dataset", "valid_dataset"])
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.debug = debug

        # ── VAE part ──────────────────────────────
        self.ae = AutoEncoder()
        with torch.no_grad():
            self.latent_dim = self.ae.encode(torch.ones(1, 3, 256, 256)).shape[1]
        self.register_buffer("latent_scale_factor", torch.tensor(latent_scale_factor))

        # ── Diffusion‑Generator (G) ───────────────
        self.model = DenoisingDiffusionConditionalProcess(
            generated_channels=self.latent_dim,
            condition_channels=self.latent_dim,
            num_timesteps=num_timesteps,
        )

        # ── PatchGAN Discriminator (D) ────────────
        self.discriminator = NLayerDiscriminator(
            input_nc=3 * self.latent_dim,  # cond + noisy + t_map
            ndf=64,
            n_layers=3,
            norm_layer=nn.BatchNorm2d,     # <<< FIX: use BatchNorm2d to avoid 1×1 InstanceNorm crash
        )

        # time embedding etc. (unchanged)
        self.time_emb = nn.Embedding(num_timesteps, self.latent_dim)
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

        # debug hooks (unchanged)
        if self.debug:
            self.discriminator.register_forward_hook(lambda m, i, o: _stat("D.out", o))
            self.model.model.register_forward_hook(lambda m, i, o: _stat("UNet.out", o))
            print("[Debug] hooks for STAT registered ✔")
            print(f"[Debug] latent_dim = {self.latent_dim}")
            tot_G = sum(p.numel() for p in self.model.parameters())
            tot_D = sum(p.numel() for p in self.discriminator.parameters())
            print(f"[Debug] Param‑count  G={tot_G/1e6:.2f}M  D={tot_D/1e6:.2f}M")
    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _to_latent(self, img: torch.Tensor):
        return (
            self.ae.encode(self._img2range(img)).detach() * self.latent_scale_factor
        )

    @staticmethod
    def _img2range(x):
        return x.clamp(0, 1) * 2 - 1

    def _range2img(self, x):
        return (x + 1) / 2

    # ---------------------------------------------------------------------
    # Lightning required
    # ---------------------------------------------------------------------
    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [opt_G, opt_D]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    # ---------------------------------------------------------------------
    # Training step
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        opt_G, opt_D = self.optimizers()
        self.step_counter += 1

        cond_img, tgt_img = batch
        device = cond_img.device

        #  encode to latent space (no grad)
        with torch.no_grad():
            cond_lat = self._to_latent(cond_img.to(device))
            real_lat = self._to_latent(tgt_img.to(device))

        # ---------------- Discriminator update ----------------
        b = real_lat.size(0)
        t = torch.randint(0, self.model.forward_process.num_timesteps - 1, (b,), device=device)

        x_t   = self.model.forward_process(real_lat, t, return_noise=False)
        x_tp1 = self.model.forward_process(real_lat, t + 1, return_noise=False)

        real_noisy = x_t.detach().requires_grad_(True)

        with torch.no_grad():
            eps_hat = self.model.model(torch.cat([x_tp1, cond_lat], 1), t)
            sqrt1m = self.model.forward_process.sqrt_one_minus_alphas_cumprod[t + 1].view(-1, 1, 1, 1)
            sqrtab = self.model.forward_process.sqrt_alphas_cumprod[t + 1].view(-1, 1, 1, 1)
            x0_hat = (x_tp1 - sqrt1m * eps_hat) / sqrtab
            mean, var = self.model.forward_process.get_posterior_mean_variance(x0_hat, x_tp1, t)
            x_fake = mean + torch.sqrt(var) * torch.randn_like(var)

        fake_noisy = x_fake.detach().requires_grad_(True)

        # concat condition + noisy + t_map
        t_map = self.time_emb(t).view(b, self.latent_dim, 1, 1).expand(-1, -1, *real_noisy.shape[2:])
        real_input = torch.cat([cond_lat, real_noisy, t_map], dim=1)
        fake_input = torch.cat([cond_lat, fake_noisy, t_map], dim=1)

        logits_real = self.discriminator(real_input)
        logits_fake = self.discriminator(fake_input)

        rel_logits_D = logits_real - logits_fake
        loss_D_adv = F.softplus(-rel_logits_D).mean()

        # R1 / R2
        grad_real = torch.autograd.grad((logits_real.sum()), real_noisy, create_graph=True)[0]
        grad_fake = torch.autograd.grad((logits_fake.sum()), fake_noisy, create_graph=True)[0]
        r1 = grad_real.pow(2).reshape(b, -1).sum(1).mean()
        r2 = grad_fake.pow(2).reshape(b, -1).sum(1).mean()

        loss_D = loss_D_adv + 0.5 * self.hparams.gamma_r1 * (r1 + r2)

        opt_D.zero_grad()
        self.manual_backward(loss_D)
        opt_D.step()

        # debug prints -----------------------------------------------------
        if self.debug and batch_idx % 50 == 0:
            _stat("logits_real", logits_real)
            _stat("logits_fake", logits_fake)
            print(
                f"[D] step={int(self.step_counter.item())}  "
                f"loss_D={loss_D.item():.4f}  r1={r1.item():.4f}  r2={r2.item():.4f}"
            )
            print(f"       grad norm D = {_grad_norm(self.discriminator):.4f}")             # <<< MOD

        # ---------------- Generator update --------------------
        # diffusion loss
        t_G = torch.randint(0, self.model.forward_process.num_timesteps - 1, (b,), device=device)
        x_noisy, eps = self.model.forward_process(real_lat, t_G, return_noise=True)
        eps_hat_G = self.model.model(torch.cat([x_noisy, cond_lat], 1), t_G)
        loss_diff = F.mse_loss(eps_hat_G, eps)

        # GAN loss (relativistic)
        with torch.no_grad():
            logits_real_G = self.discriminator(real_input)
        logits_fake_G = self.discriminator(fake_input.detach())
        rel_logits_G = logits_fake_G - logits_real_G
        loss_G_adv = F.softplus(-rel_logits_G).mean()

        loss_G = loss_diff + loss_G_adv

        opt_G.zero_grad()
        self.manual_backward(loss_G)
        opt_G.step()

        if self.debug and batch_idx % 50 == 0:
            _stat("eps_hat_G", eps_hat_G)
            print(
                f"[G] step={int(self.step_counter.item())} diff={loss_diff.item():.4f} "
                f"gan={loss_G_adv.item():.4f}  total={loss_G.item():.4f}"
            )
            print(f"       grad norm G = {_grad_norm(self.model):.4f}\n")            # <<< MOD

        # log to Lightning
        self.log_dict(
            {
                "loss_D": loss_D,
                "loss_G": loss_G,
                "loss_diff": loss_diff,
                "loss_G_adv": loss_G_adv,
                "loss_D_adv": loss_D_adv,
            },
            on_step=True,
            prog_bar=True,
        )

    # ---------------------------------------------------------------------
    # Validation (diffusion MSE only) – optional
    # ---------------------------------------------------------------------
    def val_dataloader(self):
        if self.valid_dataset is None:
            return None
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        cond_img, tgt_img = batch
        cond_lat = self._to_latent(cond_img.to(self.device))
        real_lat = self._to_latent(tgt_img.to(self.device))

        b = real_lat.size(0)
        t = torch.randint(0, self.model.forward_process.num_timesteps, (b,), device=self.device)
        x_noisy, eps = self.model.forward_process(real_lat, t, return_noise=True)
        eps_hat = self.model.model(torch.cat([x_noisy, cond_lat], 1), t)
        diff_loss = F.mse_loss(eps_hat, eps)
        self.log("val_diff_loss", diff_loss, prog_bar=True)
        return diff_loss
