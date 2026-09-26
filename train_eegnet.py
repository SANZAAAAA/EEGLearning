#!/usr/bin/env python
"""EEGNet 脑电解码主程序。

用法示例
--------
# 下载数据 + 5 折交叉验证训练 10 名被试 (默认)
python train_eegnet.py

# 先冒烟测试: 1 名被试, 每类 10 个试次, 20 个 epoch
python train_eegnet.py --subjects 1 --max-per-class 10 --epochs 20 --cv 2

# 完整实验: 109 名被试, 带经典脑电分析图
python train_eegnet.py --subjects 1-109 --cv 5 --analysis

# 只下载数据
python train_eegnet.py --subjects 1-20 --download-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from eegnet.analysis import analyze_subject
from eegnet.config import Config, parse_int_list
from eegnet.data import download_dataset, load_subject
from eegnet.engine import evaluate_subject, resolve_device
from eegnet.plots import plot_confusion_matrix, plot_subject_summary, plot_training_curves


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EEGNet 脑电解码 (PhysioNet EEG Motor Movement/Imagery, 想象左右手)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="数据缓存目录")
    p.add_argument("--subjects", type=str, default="1-10", help="被试编号, 如 1-10 或 1,3,5")
    p.add_argument("--runs", type=str, default="4,8,12", help="使用哪些 run (4/8/12 = 想象左右手)")
    p.add_argument("--tmin", type=float, default=0.0, help="epoch 起点 (秒, 相对提示符)")
    p.add_argument("--tmax", type=float, default=4.0, help="epoch 终点 (秒)")
    p.add_argument("--l-freq", type=float, default=0.5, help="带通下限 Hz")
    p.add_argument("--h-freq", type=float, default=40.0, help="带通上限 Hz")
    p.add_argument("--reference", choices=["average", "none"], default="average")
    p.add_argument("--normalization", choices=["per_trial", "global", "none"], default="per_trial")
    p.add_argument("--max-per-class", type=int, default=None, help="每类最多试次数 (调试用)")
    # 划分
    p.add_argument("--cv", type=int, default=5, help="交叉验证折数; 1 = 单次留出划分")
    p.add_argument("--test-size", type=float, default=0.2, help="cv=1 时的测试集比例")
    p.add_argument("--val-size", type=float, default=0.2, help="训练集内部验证集比例")
    p.add_argument("--seed", type=int, default=42)
    # 模型
    p.add_argument("--F1", type=int, default=8)
    p.add_argument("--D", type=int, default=2)
    p.add_argument("--F2", type=int, default=16)
    p.add_argument("--kernel-seconds", type=float, default=0.5, help="时间卷积核时长 (秒)")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--pool1", type=int, default=4)
    p.add_argument("--pool2", type=int, default=8)
    # 训练
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--augment", action="store_true", help="启用时间平移 + 噪声增强")
    p.add_argument("--class-weights", action="store_true", help="类别加权交叉熵")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    # 输出
    p.add_argument("--out", type=Path, default=Path("results"), help="结果根目录")
    p.add_argument("--run-name", type=str, default=None, help="本次运行的子目录名")
    p.add_argument("--analysis", action="store_true", help="额外输出 PSD / 频带功率地形图")
    p.add_argument("--analysis-only", action="store_true",
                   help="只跑经典脑电分析 (PSD/地形图), 不重新训练")
    p.add_argument("--download-only", action="store_true", help="只下载数据")
    return p


def config_from_args(args) -> Config:
    cfg = Config()
    cfg.data_dir = Path(args.data_dir)
    cfg.subjects = parse_int_list(args.subjects)
    cfg.runs = parse_int_list(args.runs)
    cfg.tmin, cfg.tmax = args.tmin, args.tmax
    cfg.l_freq, cfg.h_freq = args.l_freq, args.h_freq
    cfg.reference = args.reference
    cfg.normalization = args.normalization
    cfg.max_per_class = args.max_per_class
    cfg.cv = args.cv
    cfg.test_size, cfg.val_size = args.test_size, args.val_size
    cfg.seed = args.seed
    cfg.F1, cfg.D, cfg.F2 = args.F1, args.D, args.F2
    cfg.kernel_seconds = args.kernel_seconds
    cfg.dropout = args.dropout
    cfg.pool1, cfg.pool2 = args.pool1, args.pool2
    cfg.epochs, cfg.batch_size = args.epochs, args.batch_size
    cfg.lr, cfg.weight_decay = args.lr, args.weight_decay
    cfg.optimizer, cfg.patience = args.optimizer, args.patience
    cfg.augment, cfg.class_weights = args.augment, args.class_weights
    cfg.device = args.device
    cfg.out_dir = Path(args.out)
    cfg.run_name = args.run_name
    cfg.analysis, cfg.download_only = args.analysis, args.download_only
    if args.analysis_only:
        cfg.analysis = True
    return cfg


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.out_dir) / (cfg.run_name or f"run_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    log("=" * 72)
    log(f"EEGNet 脑电解码  |  开始时间 {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 72)
    log(f"配置: {cfg.describe()}")
    log(f"结果目录: {run_dir.resolve()}")

    # ---- 1. 下载数据 ----
    log("\n[1/4] 下载/校验数据 ...")
    for subj in cfg.subjects:
        paths = download_dataset([subj], cfg.runs, cfg.data_dir)[subj]
        log(f"  sub-{subj:03d}: {len(paths)} 个文件 -> {paths[0].parent}")

    if cfg.download_only:
        log(f"\n[完成] 仅下载。数据位于 {(Path(cfg.data_dir) / 'MNE-eegbci-data').resolve()}")
        log(f"耗时 {time.time() - t_start:.1f}s")
        return 0

    # ---- 只做经典脑电分析 ----
    if args.analysis_only:
        log("\n[分析] 输出 PSD / 频带功率地形图 / 侧化指数 ...")
        analysis_dir = run_dir / "analysis"
        analysis_results = []
        for subj in cfg.subjects:
            try:
                data = load_subject(cfg, subj, keep_epochs=True)
                analysis_results.append(analyze_subject(data, analysis_dir))
            except Exception as exc:                   # noqa: BLE001
                log(f"  [skip] sub-{subj:03d} 分析失败: {exc}")
        (analysis_dir / "summary_analysis.json").write_text(
            json.dumps(analysis_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(f"\n[完成] 分析结果: {analysis_dir.resolve()}")
        log(f"耗时 {time.time() - t_start:.1f}s")
        return 0

    device = resolve_device(cfg.device)
    log(f"\n[2/4] 计算设备: {device}")
    cfg.dump_json(run_dir / "config.json")

    # ---- 2. 逐被试加载 + 训练 ----
    log("\n[3/4] 逐被试训练与评估 ...")
    rows, all_metrics, all_analysis = [], [], []
    for subj in cfg.subjects:
        log(f"\n  ── sub-{subj:03d} " + "─" * 50)
        try:
            data = load_subject(cfg, subj, keep_epochs=cfg.analysis)
        except Exception as exc:                       # 数据缺失/异常时跳过
            log(f"  [skip] 加载失败: {exc}")
            continue

        if data.epochs is not None and len(np.unique(data.y)) < 2:
            log("  [skip] 类别不足两类")
            continue

        subj_dir = run_dir / f"sub-{subj:03d}"
        try:
            metrics = evaluate_subject(cfg, data, subj_dir, device)
        except Exception as exc:
            log(f"  [skip] 训练失败: {exc}")
            continue

        all_metrics.append(metrics)
        log(f"  ✓ 准确率 {metrics['accuracy']:.4f} | 平衡准确率 {metrics['balanced_accuracy']:.4f} "
            f"| kappa {metrics['kappa']:.3f} | AUC {metrics['auc'] if metrics['auc'] is None else round(metrics['auc'], 3)} "
            f"| 折均值 {metrics['fold_accuracy_mean']:.4f}±{metrics['fold_accuracy_std']:.4f} "
            f"| 用时 {metrics['elapsed_sec']:.1f}s")

        plot_training_curves(
            json.loads((subj_dir / "history.json").read_text(encoding="utf-8")),
            subj_dir / "training_curves.png",
            title=f"sub-{subj:03d} EEGNet training",
        )
        plot_confusion_matrix(
            metrics["confusion_matrix"], data.plot_class_names,
            subj_dir / "confusion_matrix.png",
            title=f"sub-{subj:03d} (acc={metrics['accuracy']:.3f})",
        )

        if cfg.analysis and data.epochs is not None:
            try:
                all_analysis.append(analyze_subject(data, run_dir / "analysis"))
            except Exception as exc:
                log(f"  [warn] 分析失败: {exc}")

        rows.append({
            "subject": metrics["subject"],
            "n_trials": metrics["n_trials"],
            "n_channels": metrics["n_channels"],
            "n_times": metrics["n_times"],
            "cv_folds": metrics["cv_folds"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "kappa": metrics["kappa"],
            "auc": metrics["auc"],
            "fold_accuracy_mean": metrics["fold_accuracy_mean"],
            "fold_accuracy_std": metrics["fold_accuracy_std"],
            "elapsed_sec": metrics["elapsed_sec"],
        })

    if not rows:
        log("\n[错误] 没有任何被试成功完成训练。")
        return 1

    # ---- 3. 汇总 ----
    log("\n[4/4] 汇总结果 ...")
    df = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    df.to_csv(run_dir / "summary.csv", index=False)

    overall = {
        "n_subjects": int(len(df)),
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std(ddof=1)) if len(df) > 1 else 0.0,
        "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
        "kappa_mean": float(df["kappa"].mean()),
        "auc_mean": float(df["auc"].dropna().mean()) if df["auc"].notna().any() else None,
        "fold_accuracy_mean_of_means": float(df["fold_accuracy_mean"].mean()),
        "min_accuracy": float(df["accuracy"].min()),
        "max_accuracy": float(df["accuracy"].max()),
        "elapsed_sec_total": float(time.time() - t_start),
    }
    (run_dir / "summary.json").write_text(
        json.dumps({"overall": overall, "per_subject": all_metrics,
                    "config": cfg.to_dict(), "analysis": all_analysis},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_subject_summary(df, run_dir / "summary_accuracy.png")

    log("\n" + "=" * 72)
    log("被试级结果 (交叉验证):")
    log(df[["subject", "accuracy", "balanced_accuracy", "kappa", "auc"]].to_string(index=False))
    log("")
    log(f"被试数            : {overall['n_subjects']}")
    log(f"平均准确率        : {overall['accuracy_mean']:.4f} ± {overall['accuracy_std']:.4f} "
        f"(随机水平 0.5000)")
    log(f"平均平衡准确率    : {overall['balanced_accuracy_mean']:.4f}")
    log(f"平均 Cohen's kappa: {overall['kappa_mean']:.4f}")
    if overall["auc_mean"] is not None:
        log(f"平均 AUC          : {overall['auc_mean']:.4f}")
    log(f"总耗时            : {overall['elapsed_sec_total']:.1f}s")
    log(f"结果目录          : {run_dir.resolve()}")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
