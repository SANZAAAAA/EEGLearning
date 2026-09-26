"""经典脑电分析: 功率谱、频带功率地形图、ERD/ERS 侧化。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import mne                        # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

from .data import SubjectData     # noqa: E402

BANDS = {
    "mu (8-13 Hz)": (8.0, 13.0),
    "beta (13-30 Hz)": (13.0, 30.0),
}


def _epochs_with_clean_names(data: SubjectData):
    """返回 (epochs, cmap_ok), epochs 的导联名已可用于 topomap。"""
    epochs = data.epochs
    if epochs is None:
        raise ValueError("需要 keep_epochs=True 才能做分析")
    montage = mne.channels.make_standard_montage("standard_1005")
    known = set(montage.ch_names)
    ok = all(ch in known for ch in epochs.ch_names)
    return epochs, ok


def _topomap(values, info, ax, vlim):
    """画地形图并返回可传给 colorbar 的对象。

    不同 MNE 版本的 ``plot_topomap`` 返回值不一样 (旧版返回 AxesImage,
    新版返回 (image, contour) 元组), 这里统一处理。
    """
    res = mne.viz.plot_topomap(
        values, info, axes=ax, show=False, cmap="RdBu_r",
        vlim=vlim, contours=0, sensors=True,
    )
    if isinstance(res, (tuple, list)):
        for item in res:
            if hasattr(item, "cmap"):
                return item
        return res[0]
    return res


def _add_colorbar(fig, mappable, ax) -> None:
    try:
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    except Exception:            # 只为出图美观, 失败不影响分析结果
        pass


def plot_psd(data: SubjectData, out_path: Path) -> None:
    """两类试次的平均功率谱密度 (Welch)。"""
    epochs, _ = _epochs_with_clean_names(data)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    spec = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose="ERROR")
    psds = spec.get_data()                       # (trials, channels, freqs)
    freqs = spec.freqs
    for cls in np.unique(data.y):
        psd = psds[data.y == cls].mean(axis=(0, 1)) * 1e12   # uV^2/Hz
        ax.plot(freqs, psd, lw=1.4, label=data.plot_class_names[cls])
    for lo, hi in BANDS.values():
        ax.axvspan(lo, hi, color="orange", alpha=0.08)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (uV^2/Hz)")
    ax.set_title(f"Mean PSD across 64 channels (sub-{data.subject:03d})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def band_power_maps(data: SubjectData, out_path: Path) -> dict:
    """mu / beta 频带功率地形图, 以及两类的差异图 (dB)。"""
    epochs, cmap_ok = _epochs_with_clean_names(data)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_class = len(data.class_names)
    n_bands = len(BANDS)
    fig, axes = plt.subplots(n_class + 1, n_bands, figsize=(5.4, 2.4 * (n_class + 1)))
    axes = np.atleast_2d(axes)
    info = epochs.info

    summary: dict = {}
    band_values = {}
    for b_idx, (band, (lo, hi)) in enumerate(BANDS.items()):
        filt = epochs.copy().filter(lo, hi, fir_design="firwin", verbose="ERROR")
        power = (filt.get_data() ** 2).mean(axis=2)          # (trials, channels) uV^2
        per_class = np.stack([power[data.y == c].mean(axis=0) for c in range(n_class)])
        band_values[band] = per_class
        vmax = float(np.percentile(per_class, 97.5))
        for c in range(n_class):
            ax = axes[c, b_idx]
            im = _topomap(per_class[c], info, ax, (0, vmax))
            ax.set_title(f"{data.plot_class_names[c]}\n{band}", fontsize=8)
            _add_colorbar(fig, im, ax)
            summary[f"{band}|{data.class_names[c]}"] = float(per_class[c].mean())

        ratio_db = 10 * np.log10((per_class[0] + 1e-12) / (per_class[1] + 1e-12))
        ax = axes[n_class, b_idx]
        lim = float(np.percentile(np.abs(ratio_db), 98))
        im = _topomap(ratio_db, info, ax, (-lim, lim))
        ax.set_title(f"T1 - T2 (dB)\n{band}", fontsize=8)
        _add_colorbar(fig, im, ax)
        summary[f"{band}|T1-T2_dB_mean"] = float(ratio_db.mean())

    if not cmap_ok:
        fig.suptitle("channel positions partially unresolved", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return summary


def lateralization_index(data: SubjectData) -> pd.DataFrame:
    """C3/C4 (运动皮层) 的 mu 频带功率侧化指数。

    LI = (C4 - C3) / (C4 + C3)。想象左手时右侧皮层 (C4) 的 mu 功率下降,
    因此 T1 条件通常 LI > 0。
    """
    epochs, _ = _epochs_with_clean_names(data)
    lo, hi = BANDS["mu (8-13 Hz)"]
    filt = epochs.copy().filter(lo, hi, fir_design="firwin", verbose="ERROR")
    power = (filt.get_data() ** 2).mean(axis=2)              # (trials, channels)
    ch = {name: i for i, name in enumerate(filt.ch_names)}
    if "C3" not in ch or "C4" not in ch:
        return pd.DataFrame()
    p3, p4 = power[:, ch["C3"]], power[:, ch["C4"]]
    li = (p4 - p3) / (p4 + p3 + 1e-12)
    rows = []
    for cls in np.unique(data.y):
        vals = li[data.y == cls]
        rows.append({
            "subject": data.subject,
            "class": data.class_names[cls],
            "n_trials": int(len(vals)),
            "C3_power": float(p3[data.y == cls].mean()),
            "C4_power": float(p4[data.y == cls].mean()),
            "LI_mean": float(vals.mean()),
            "LI_std": float(vals.std()),
        })
    return pd.DataFrame(rows)


def analyze_subject(data: SubjectData, out_dir: Path, verbose: bool = True) -> dict:
    """对一个被试跑完整分析流程, 图存到 ``out_dir``。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"sub-{data.subject:03d}"
    result: dict = {"subject": int(data.subject), "n_trials": int(data.n_trials)}

    plot_psd(data, out_dir / f"{tag}_psd.png")
    result["band_power"] = band_power_maps(data, out_dir / f"{tag}_band_power_topomaps.png")

    li = lateralization_index(data)
    if not li.empty:
        li.to_csv(out_dir / f"{tag}_lateralization.csv", index=False)
        result["lateralization_index"] = li.to_dict(orient="records")
    if verbose:
        print(f"  分析图已保存: {out_dir / (tag + '_psd.png')}")
        print(f"  分析图已保存: {out_dir / (tag + '_band_power_topomaps.png')}")
    return result
