"""EEGNet 脑电分析工具箱。

数据集: PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB, 64 通道 160 Hz)
任务:   想象左手 / 右手抓握 (T1 vs T2) 的二分类解码
模型:   EEGNet (Lawhern et al., 2018, J. Neural Eng. 15:056013)
"""

import os
from pathlib import Path

# matplotlib 默认把缓存写到 ~/.matplotlib, 可能不可写; 指向项目内目录
_project_root = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".mplconfig"))

from .config import Config, parse_int_list  # noqa: E402
from .model import EEGNet, build_model  # noqa: E402

__all__ = ["Config", "parse_int_list", "EEGNet", "build_model"]
__version__ = "0.1.0"
