"""数据下载与预处理。

数据集: PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)
    - 109 名被试, 64 导联, 160 Hz 采样率
    - 每个 run 交替呈现 T1 / T2 提示符, 每次 15 个试次
    - run 4, 8, 12: 想象左手 (T1) / 右手 (T2) 抓握  <- 本项目使用

下载由 MNE 自动完成 (无需注册), 文件缓存在 ``data/MNE-eegbci-data`` 下。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MI_RUNS = (4, 8, 12)          # 想象左右手抓握
ME_RUNS = (3, 7, 11)          # 实际左右手动作

RUN_INFO = {
    1: "基线, 睁眼",
    2: "基线, 闭眼",
    3: "实际动作: 左手/右手握拳 (T1/T2)",
    4: "想象动作: 左手/右手握拳 (T1/T2)",
    5: "实际动作: 双手/双脚 (T1/T2)",
    6: "想象动作: 双手/双脚 (T1/T2)",
    7: "实际动作: 左手/右手握拳 (T1/T2)",
    8: "想象动作: 左手/右手握拳 (T1/T2)",
    9: "实际动作: 双手/双脚 (T1/T2)",
    10: "想象动作: 双手/双脚 (T1/T2)",
    11: "实际动作: 左手/右手握拳 (T1/T2)",
    12: "想象动作: 左手/右手握拳 (T1/T2)",
    13: "实际动作: 双手/双脚 (T1/T2)",
    14: "想象动作: 双手/双脚 (T1/T2)",
}

CLASS_NAMES = ["T1 (想象左手)", "T2 (想象右手)"]


def _mne():
    """延迟导入 MNE, 并关闭冗余日志。"""
    import mne

    mne.set_log_level("ERROR")
    return mne


# --------------------------------------------------------------------------- #
# 1. 下载
# --------------------------------------------------------------------------- #
def download_subject(subject: int, runs, data_dir: Path, force: bool = False) -> list[Path]:
    """下载某个被试的指定 run, 返回 EDF 文件路径列表 (顺序与 runs 一致)。"""
    from mne.datasets import eegbci

    runs = list(runs)
    files = eegbci.load_data(
        subjects=[int(subject)],
        runs=runs,
        path=str(Path(data_dir)),
        force_update=force,
        update_path=False,
        verbose="ERROR",
    )
    return [Path(f) for f in files]


def download_dataset(subjects, runs, data_dir: Path, force: bool = False) -> dict[int, list[Path]]:
    """批量下载, 返回 {被试: [文件路径, ...]}。"""
    out: dict[int, list[Path]] = {}
    for subj in subjects:
        out[int(subj)] = download_subject(int(subj), runs, data_dir, force=force)
    return out


# --------------------------------------------------------------------------- #
# 2. 导联名标准化
# --------------------------------------------------------------------------- #
def _standard_name(name: str, known: set[str]) -> str | None:
    """把 EEGMMIDB 的 'Fc5.' 这类导联名映射到 MNE 标准名 'FC5'。"""
    base = name.strip().strip(".").strip()
    candidates = [
        base,
        base.upper(),
        base.capitalize(),
        base.upper().replace("FP", "Fp"),
    ]
    if base.lower().endswith("z"):
        candidates.append(base[:-1].upper() + "z")        # Fcz -> FCz
    for cand in candidates:
        if cand in known:
            return cand
    return None


def _prepare_channels(raw):
    """统一导联名并挂载标准 10-05 电极位置 (用于画地形图)。"""
    mne = _mne()
    montage = mne.channels.make_standard_montage("standard_1005")
    known = set(montage.ch_names)

    mapping, unresolved = {}, []
    for ch in raw.ch_names:
        new = _standard_name(ch, known)
        if new is None:
            unresolved.append(ch)
        elif new != ch:
            mapping[ch] = new
    if mapping:
        raw.rename_channels(mapping)
    raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
    return unresolved


# --------------------------------------------------------------------------- #
# 3. 被试数据
# --------------------------------------------------------------------------- #
@dataclass
class SubjectData:
    """一个被试的全部试次。"""

    subject: int
    X: np.ndarray                 # (n_trials, n_channels, n_times)
    y: np.ndarray                 # (n_trials,) 0/1
    sfreq: float
    ch_names: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    plot_class_names: list[str] = field(default_factory=lambda: ["T1", "T2"])
    epochs: object | None = None  # mne.Epochs, 仅在需要画图时保留
    unresolved_channels: list[str] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        return int(self.X.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.X.shape)  # type: ignore[return-value]

    def summary(self) -> str:
        counts = np.bincount(self.y, minlength=len(self.class_names))
        return (
            f"sub-{self.subject:03d}: X={self.shape} (试次, 导联, 时间点), "
            f"sfreq={self.sfreq:g}Hz, 类别分布={dict(zip(self.class_names, counts.tolist()))}"
        )


def load_subject(
    cfg,
    subject: int,
    keep_epochs: bool = False,
    verbose: bool = True,
) -> SubjectData:
    """下载 (如缺失) 并预处理一个被试, 返回 ``SubjectData``。"""
    mne = _mne()
    from mne.datasets import eegbci

    files = download_subject(subject, cfg.runs, cfg.data_dir)
    raws = []
    for f in files:
        raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
        eegbci.standardize(raw)
        raws.append(raw)

    # 统一导联名: eegbci.standardize 会给名字加 '.' 后缀
    raw = mne.concatenate_raws(raws, verbose="ERROR")
    unresolved = _prepare_channels(raw)
    if verbose and unresolved:
        print(f"  [warn] 未匹配到标准位置的导联: {unresolved}")

    raw.pick("eeg", verbose="ERROR")
    if cfg.reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.filter(cfg.l_freq, cfg.h_freq, fir_design="firwin", verbose="ERROR")

    # 事件: T1 / T2 分别是两类提示符
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    wanted = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
    if len(wanted) < 2:
        raise RuntimeError(f"sub-{subject}: 未找到 T1/T2 标注, 请检查 runs={cfg.runs}")
    ordered = sorted(wanted)                     # ['T1', 'T2']

    epochs = mne.Epochs(
        raw,
        events,
        event_id={k: wanted[k] for k in ordered},
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=None,
        preload=True,
        proj=False,
        reject=None,
        verbose="ERROR",
    )

    X = epochs.get_data(copy=True).astype(np.float32)   # (n, C, T)
    codes = epochs.events[:, 2]
    code_to_label = {epochs.event_id[name]: i for i, name in enumerate(ordered)}
    y = np.array([code_to_label[c] for c in codes], dtype=np.int64)

    # 可选: 每个类别下采样 (快速调试)
    if cfg.max_per_class is not None:
        keep = []
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)[: cfg.max_per_class]
            keep.extend(idx.tolist())
        keep = np.sort(np.array(keep, dtype=int))
        X, y = X[keep], y[keep]
        epochs = epochs[keep]

    data = SubjectData(
        subject=int(subject),
        X=X,
        y=y,
        sfreq=float(raw.info["sfreq"]),
        ch_names=list(raw.ch_names),
        class_names=[f"{n} (想象{'左' if n == 'T1' else '右'}手)" for n in ordered],
        # 图里用纯 ASCII 标签, 避免中文字体缺字形
        plot_class_names=[f"{n} ({'left' if n == 'T1' else 'right'} hand)" for n in ordered],
        epochs=epochs if keep_epochs else None,
        unresolved_channels=list(unresolved),
    )
    if verbose:
        print("  " + data.summary())
    return data


# --------------------------------------------------------------------------- #
# 4. 归一化
# --------------------------------------------------------------------------- #
def fit_normalizer(X: np.ndarray):
    """按通道统计均值和标准差 (只在训练集上调用, 避免信息泄漏)。"""
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalize(X: np.ndarray, mode: str, stats: dict | None = None) -> tuple[np.ndarray, dict | None]:
    """归一化。

    mode
    ----
    ``per_trial`` : 每个试次单独做 z-score (最常用, 不需要训练集统计量)
    ``global``    : 用训练集的通道均值/标准差归一化
    ``none``      : 不处理
    """
    X = np.asarray(X, dtype=np.float32)
    if mode == "none":
        return X, None
    if mode == "per_trial":
        mean = X.mean(axis=(1, 2), keepdims=True)
        std = X.std(axis=(1, 2), keepdims=True) + 1e-6
        return ((X - mean) / std).astype(np.float32), None
    if mode == "global":
        stats = fit_normalizer(X) if stats is None else stats
        return ((X - stats["mean"]) / stats["std"]).astype(np.float32), stats
    raise ValueError(f"未知的归一化方式: {mode!r}")
