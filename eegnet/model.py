"""EEGNet 模型实现 (PyTorch)。

参考:
    Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ (2018).
    "EEGNet: a compact convolutional neural network for EEG-based
    brain-computer interfaces." Journal of Neural Engineering 15(5): 056013.

网络结构 (两块):
    Block1  temporal conv (1, k) -> depthwise spatial conv (C, 1) -> BN -> ELU
            -> average pool -> dropout
    Block2  depthwise temporal conv (1, k/4) -> pointwise conv (1, 1) -> BN -> ELU
            -> average pool -> dropout
    Head    flatten -> linear -> (logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet。

    参数
    ----
    n_channels : 电极数 C
    n_times    : 每个试次的时间点数 T
    n_classes  : 分类数
    F1         : 时间卷积的滤波器数量
    D          : 每个时间滤波器对应的空间滤波器个数 (depth multiplier)
    F2         : 逐点卷积的输出通道数
    kernel_length : 时间卷积核长度 (采样点数)
    dropout    : dropout 比例
    pool1/pool2: 两块的平均池化下采样倍数
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 80,
        dropout: float = 0.5,
        pool1: int = 4,
        pool2: int = 8,
    ) -> None:
        super().__init__()
        if kernel_length < 4:
            raise ValueError("kernel_length 至少为 4")

        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.F1, self.D, self.F2 = int(F1), int(D), int(F2)
        self.kernel_length = int(kernel_length)
        self.dropout_p = float(dropout)
        self.pool1, self.pool2 = int(pool1), int(pool2)

        pad1 = self.kernel_length // 2          # 'same' 填充
        k2 = max(2, self.kernel_length // 4)
        pad2 = k2 // 2

        self.block1 = nn.Sequential(
            # 时间卷积: 每个通道独立学习频率滤波器
            nn.Conv2d(1, self.F1, (1, self.kernel_length), padding=(0, pad1), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, eps=1e-3, affine=True),
            # 空间卷积 (depthwise): 把 C 个电极压成 1, 每个滤波器学一个空间模式
            nn.Conv2d(self.F1, self.F1 * self.D, (self.n_channels, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, eps=1e-3, affine=True),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool1)),
            nn.Dropout(self.dropout_p),
        )

        self.block2 = nn.Sequential(
            # 可分离卷积: 深度方向 (1, k/4) + 逐点 (1, 1)
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, k2), padding=(0, pad2),
                      groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2, momentum=0.01, eps=1e-3, affine=True),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool2)),
            nn.Dropout(self.dropout_p),
        )

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(self._infer_feature_dim(), self.n_classes)
        self.reset_parameters()

    # ------------------------------------------------------------------ #
    def _infer_feature_dim(self) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_times)
            feats = self.block2(self.block1(dummy))
        if was_training:
            self.train()
        return int(feats.flatten(1).shape[1])

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                       # (N, C, T) -> (N, 1, C, T)
            x = x.unsqueeze(1)
        x = self.block2(self.block1(x))
        return self.classifier(self.flatten(x))

    # ------------------------------------------------------------------ #
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return (
            f"EEGNet(C={self.n_channels}, T={self.n_times}, classes={self.n_classes}, "
            f"F1={self.F1}, D={self.D}, F2={self.F2}, kernel={self.kernel_length}, "
            f"dropout={self.dropout_p}, params={self.n_parameters})"
        )


def build_model(cfg, n_channels: int, n_times: int, n_classes: int, sfreq: float) -> EEGNet:
    """按配置构建 EEGNet, 并把 ``kernel_seconds`` 换算成采样点数。"""
    kernel_length = int(round(cfg.kernel_seconds * float(sfreq)))
    kernel_length -= kernel_length % 2          # 保证是偶数
    kernel_length = max(4, kernel_length)
    return EEGNet(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        F1=cfg.F1,
        D=cfg.D,
        F2=cfg.F2,
        kernel_length=kernel_length,
        dropout=cfg.dropout,
        pool1=cfg.pool1,
        pool2=cfg.pool2,
    )
