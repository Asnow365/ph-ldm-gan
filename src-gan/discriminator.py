# 文件名：discriminator_simple.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoidal_timestep_embedding(timesteps, embedding_dim: int):
    """
    简单实现：将整型 time step 转成一个与 Transformer 类似的正弦余弦 embedding。
    timesteps: tensor of shape (B,), dtype=torch.long
    返回：(B, embedding_dim) 的浮点张量
    """
    half_dim = embedding_dim // 2
    exp_term = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / (half_dim - 1)
    exp_term = torch.exp(exp_term)  # (half_dim,)
    # timesteps.float()[..., None]: (B,1)  exp_term[None, :]:(1,half_dim) -> (B, half_dim)
    emb = timesteps.float().unsqueeze(1) * exp_term.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, embedding_dim) 或 (B, embedding_dim-1)＋1 裁剪
    if embedding_dim % 2 == 1:  # 如果是奇数维，就补一个 0
        emb = F.pad(emb, (0, 1))
    return emb  # (B, embedding_dim)


class TimeEmbedding(nn.Module):
    """
    将时间步 embedding 通过两层 MLP，输出到指定维度。
    """
    def __init__(self, embedding_dim: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, t: torch.Tensor):
        """
        t: (B,) long 型时间步
        返回: (B, out_dim)
        """
        temb = get_sinusoidal_timestep_embedding(t, self.lin1.in_features)  # (B, embedding_dim)
        temb = self.act(self.lin1(temb))
        temb = self.lin2(temb)
        return temb  # (B, out_dim)


class DownBlock(nn.Module):
    """
    下采样卷积块：Conv -> 加入 time embedding -> ReLU -> Conv -> Skip Connection -> Optional Downsample
    """
    def __init__(self, in_ch, out_ch, t_emb_dim, downsample: bool):
        super().__init__()
        self.downsample = downsample

        # 第一层卷积
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        # 用 1×1 卷积把 time embedding 投影到 out_ch 通道
        self.time_proj = nn.Linear(t_emb_dim, out_ch)
        # 第二层卷积
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # 用于 skip connection：直接 1×1 将 in_ch 投影到 out_ch
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.act = nn.LeakyReLU(0.2)

        # 如果需要下采样，就用平均池化
        if self.downsample:
            self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        x: (B, in_ch, H, W)
        t_emb: (B, t_emb_dim)
        返回: (B, out_ch, H/2, W/2) 如果 downsample=True，否则 (B, out_ch, H, W)
        """
        h = self.act(x)
        h = self.conv1(h)  # (B, out_ch, H, W)
        # 把时间嵌入投影、加到特征里
        # t_emb: (B, out_ch) -> reshape (B, out_ch, 1, 1) -> 与 h 相加
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)
        h = self.conv2(h)  # (B, out_ch, H, W)

        # 如果需要下采样，则先池化 h 和池化 x 以保持尺寸一致
        if self.downsample:
            h = self.pool(h)          # (B, out_ch, H/2, W/2)
            x = self.pool(x)          # (B, in_ch, H/2, W/2)

        # skip connection
        skip = self.skip(x)           # (B, out_ch, H/2, W/2) 或 (B, out_ch, H, W)
        out = (h + skip) / math.sqrt(2)
        return out


class DiscriminatorSimple(nn.Module):
    """
    极简化版的 LLDGAN 判别器。接收带噪潜在 x_t 和上一阶潜在 x_{t+1}：
      - 拼接成 (B, 2*C, H, W)
      - 多层 DownBlock 提取特征
      - 可选 StdDev 统计信息
      - 最后 FC 输出一个标量判别分数
    """
    def __init__(
        self,
        latent_channels: int,   # VAE 输出潜在通道数 (一般是 4)
        t_emb_dim: int = 128,   # 时间嵌入维度
        base_ch: int = 64,      # 第一层通道数，后续倍增
        num_down: int = 3       # 下采样块数量
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.t_emb_dim = t_emb_dim

        # 时间嵌入模块
        self.time_emb = TimeEmbedding(embedding_dim=t_emb_dim, out_dim=t_emb_dim)

        # 第一层，将 (2*latent_channels) 投影到 base_ch
        self.init_conv = nn.Conv2d(2 * latent_channels, base_ch, kernel_size=1)

        # 构造多个 DownBlock，不断下采样
        ch = base_ch
        self.down_blocks = nn.ModuleList()
        for i in range(num_down):
            next_ch = ch * 2 if i > 0 else ch  # 第一层不翻倍，后面每次翻倍
            # downsample=True 表示从第二层开始做下采样
            block = DownBlock(in_ch=ch, out_ch=next_ch, t_emb_dim=t_emb_dim, downsample=(i > 0))
            self.down_blocks.append(block)
            ch = next_ch

        # 最后一层让通道数变成 ch
        self.final_conv = nn.Conv2d(ch + 1, ch, kernel_size=3, padding=1)  # +1 用于接 stddev 信息

        # FC 层输出一个标量
        self.fc = nn.Linear(ch, 1)

        # 为 StdDev 统计准备分组参数
        self.stddev_group = 4
        self.stddev_feat = 1

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x_t_minus_1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """
        x_t_minus_1: (B, latent_channels, H, W)  ← 代表上一时刻的 latent（干净或少许噪声）
        x_t:         (B, latent_channels, H, W)  ← 代表当前时刻的带噪 latent
        t:           (B,) long  时间步
        返回：   (B, 1)  判别器分数
        """
        B, C, H, W = x_t.shape

        # 1) 时间嵌入
        t_emb = self.time_emb(t)  # (B, t_emb_dim)

        # 2) 拼接输入
        inp = torch.cat([x_t_minus_1, x_t], dim=1)  # (B, 2*C, H, W)
        h = self.init_conv(inp)                     # (B, base_ch, H, W)

        # 3) 多层下采样
        for block in self.down_blocks:
            h = block(h, t_emb)  # 每次都把上一次的输出当成新的输入

        # 4) StdDev 均值统计 (给判别器添一点分布信息)
        batch, ch, hh, ww = h.shape
        group = min(batch, self.stddev_group)
        # 按 (group, -1, stddev_feat, ch/stddev_feat, hh, ww) 形状拆分
        stddev = h.view(group, -1, self.stddev_feat, ch // self.stddev_feat, hh, ww)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)        # ( -1, stddev_feat, hh, ww )
        stddev = stddev.mean([1, 2, 3], keepdim=True)                     # ( 1,1,1,1 )
        stddev = stddev.repeat(group, 1, hh, ww)                           # (group,1,hh,ww)
        h = torch.cat([h, stddev], dim=1)  # (B, ch+1, hh, ww)

        # 5) 最后一层卷积 + 激活
        h = self.act(self.final_conv(h))  # (B, ch, hh, ww)

        # 6) 全局 Pool + FC 得分
        h = h.view(B, ch, -1).sum(dim=2)  # (B, ch)  全局 sum-pool
        out = self.fc(h)                  # (B, 1)
        return out


if __name__ == "__main__":
    # 简单测试一下 shape 是否匹配
    B, latent_c, H, W = 4, 4, 16, 16
    D = DiscriminatorSimple(latent_channels=latent_c, t_emb_dim=64, base_ch=32, num_down=3)
    x_prev = torch.randn(B, latent_c, H, W)
    x_curr = torch.randn(B, latent_c, H, W)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)
    out = D(x_prev, x_curr, t)
    print("Discriminator output shape:", out.shape)  # 预期 (4, 1)
