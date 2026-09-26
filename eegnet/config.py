"""全局配置。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def parse_int_list(spec: str | list[int]) -> list[int]:
    """把 ``"1-10"`` / ``"1,3,5"`` / ``"2-4,7"`` 解析成整数列表。"""
    if isinstance(spec, (list, tuple)):
        return sorted({int(x) for x in spec})
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"无法解析被试编号: {spec!r}")
    return sorted(set(out))


@dataclass
class Config:
    """一次实验的全部超参数。"""

    # ---------------- 数据 ----------------
    data_dir: Path = Path("data")
    subjects: list[int] = field(default_factory=lambda: list(range(1, 11)))
    runs: list[int] = field(default_factory=lambda: [4, 8, 12])
    tmin: float = 0.0            # 相对提示符 (cue) 的起始时间, 秒
    tmax: float = 4.0            # 结束时间, 秒
    l_freq: float = 0.5          # 带通下限 Hz
    h_freq: float = 40.0         # 带通上限 Hz
    reference: str = "average"   # average | none
    normalization: str = "per_trial"  # per_trial | global | none
    max_per_class: int | None = None  # 每个类别最多取多少个试次 (调试用)

    # ---------------- 数据划分 ----------------
    cv: int = 5                  # >1 时用分层 K 折交叉验证
    test_size: float = 0.2       # cv<=1 时的留出测试集比例
    val_size: float = 0.2        # 从训练集里再切出的验证集比例 (早停用)
    seed: int = 42

    # ---------------- 模型 ----------------
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_seconds: float = 0.5  # 时间卷积核长度(秒), 会换算成采样点数
    dropout: float = 0.5
    pool1: int = 4
    pool2: int = 8

    # ---------------- 训练 ----------------
    epochs: int = 150
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"      # adam | adamw
    patience: int = 30           # 早停耐心值 (按验证集 loss)
    min_delta: float = 1e-4
    augment: bool = False        # 训练时做时间平移 + 高斯噪声增强
    class_weights: bool = False  # 类别加权交叉熵
    device: str = "auto"         # auto | cpu | mps | cuda

    # ---------------- 输出 ----------------
    out_dir: Path = Path("results")
    run_name: str | None = None
    analysis: bool = False       # 是否额外做 PSD / 频带功率地形图分析
    download_only: bool = False  # 只下载数据, 不训练

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["out_dir"] = str(self.out_dir)
        return d

    def dump_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def describe(self) -> str:
        return (
            f"subjects={self.subjects} runs={self.runs} "
            f"epoch=[{self.tmin}, {self.tmax}]s band=[{self.l_freq}, {self.h_freq}]Hz "
            f"ref={self.reference} norm={self.normalization} "
            f"cv={self.cv} epochs={self.epochs} lr={self.lr} "
            f"EEGNet(F1={self.F1}, D={self.D}, F2={self.F2}, dropout={self.dropout})"
        )
