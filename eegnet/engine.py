"""训练 / 评估引擎: 分折训练、早停、指标计算与结果落盘。"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from .data import SubjectData, normalize
from .model import build_model


# --------------------------------------------------------------------------- #
# 设备
# --------------------------------------------------------------------------- #
def resolve_device(spec: str = "auto") -> torch.device:
    """选择计算设备。

    注意: Apple MPS 在部分受限环境下虽然 ``is_available()`` 为真, 但实际算子会
    报 "The MPS backend is supported on macOS 14.0+", 所以这里做一次真实探测,
    失败就回退到 CPU。
    """

    def _probe(device: torch.device) -> bool:
        try:
            conv = nn.Conv2d(1, 4, (1, 8), padding=(0, 4)).to(device)
            out = conv(torch.zeros(2, 1, 4, 64, device=device))
            if device.type == "mps":
                torch.mps.synchronize()
            return out.shape[-1] == 64
        except Exception as exc:      # noqa: BLE001
            print(f"  [warn] {device} 不可用 ({exc}); 回退到 CPU")
            return False

    if spec != "auto":
        device = torch.device(spec)
        if device.type in {"mps", "cuda"} and not _probe(device):
            return torch.device("cpu")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if _probe(device):
            return device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        if _probe(device):
            return device
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 数据集 (含可选数据增强)
# --------------------------------------------------------------------------- #
class EpochDataset(Dataset):
    """(n_trials, C, T) 的试次数据集。"""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False,
                 shift_frac: float = 0.1, noise_std: float = 0.1) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        self.augment = augment
        self.shift_frac = shift_frac
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        if self.augment:
            n_times = x.shape[-1]
            max_shift = max(1, int(round(self.shift_frac * n_times)))
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift:
                x = np.roll(x, shift, axis=-1)      # 循环时间平移
            if self.noise_std > 0:
                x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
        return torch.from_numpy(x), int(self.y[idx])


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names) -> dict:
    """准确率 / 平衡准确率 / Cohen's kappa / AUC / 混淆矩阵。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_prob).argmax(axis=1).astype(int)
    n_classes = y_prob.shape[1]

    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }
    if n_classes == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            out["auc"] = None
    else:
        out["auc"] = None

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    out["per_class"] = {
        name: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
        for i, name in enumerate(class_names)
    }
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()
    return out


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    probs = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32)).to(device)
        logits = model(xb)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def _eval_loss(model: nn.Module, criterion, X: np.ndarray, y: np.ndarray,
               device: torch.device, batch_size: int = 256) -> float:
    model.eval()
    total, n = 0.0, 0
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32)).to(device)
        yb = torch.from_numpy(np.asarray(y[start:start + batch_size], dtype=np.int64)).to(device)
        loss = criterion(model(xb), yb)
        total += float(loss.item()) * len(yb)
        n += len(yb)
    return total / max(n, 1)


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
@dataclass
class FitOutput:
    model: nn.Module
    history: dict = field(default_factory=dict)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    n_epochs_run: int = 0


def fit_model(
    cfg,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sfreq: float,
    device: torch.device,
    seed: int,
    verbose: bool = False,
) -> FitOutput:
    """训练一个 EEGNet, 按验证集 loss 早停, 返回验证集最优的权重。"""
    set_seed(seed)
    n_classes = int(max(y_tr.max(), y_val.max()) + 1)
    model = build_model(cfg, X_tr.shape[1], X_tr.shape[2], n_classes, sfreq).to(device)

    train_ds = EpochDataset(X_tr, y_tr, augment=cfg.augment)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch_size, len(train_ds)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    if cfg.class_weights:
        counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
        w = len(y_tr) / (n_classes * np.maximum(counts, 1.0))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss, best_epoch, wait = float("inf"), 0, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum, correct, n_seen = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            n_seen += len(yb)

        train_loss = loss_sum / max(n_seen, 1)
        train_acc = correct / max(n_seen, 1)
        val_loss = _eval_loss(model, criterion, X_val, y_val, device)
        val_acc = float(
            (predict_proba(model, X_val, device).argmax(axis=1) == np.asarray(y_val)).mean()
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 25 == 0 or epoch == 1):
            print(f"      epoch {epoch:3d}  train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_loss < best_val_loss - cfg.min_delta:
            best_val_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                if verbose:
                    print(f"      早停于 epoch {epoch} (最优 epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return FitOutput(model=model, history=history, best_epoch=best_epoch,
                     best_val_loss=best_val_loss, n_epochs_run=len(history["train_loss"]))


# --------------------------------------------------------------------------- #
# 单个被试的完整评估
# --------------------------------------------------------------------------- #
def _normalize_split(X_tr, y_tr, X_te, mode):
    """按训练集统计量归一化 (per_trial 时两组各自独立)。"""
    if mode == "global":
        X_tr, stats = normalize(X_tr, mode)
        X_te, _ = normalize(X_te, mode, stats)
    else:
        X_tr, _ = normalize(X_tr, mode)
        X_te, _ = normalize(X_te, mode)
    return X_tr, X_te


def evaluate_subject(
    cfg,
    data: SubjectData,
    out_dir: Path,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """对单个被试做交叉验证 (``cfg.cv>1``) 或留出法评估, 结果写入 ``out_dir``。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all = data.X, data.y
    n_classes = len(data.class_names)
    min_class = int(np.bincount(y_all, minlength=n_classes).min())
    t0 = time.time()

    histories: list[dict] = []
    fold_accs: list[float] = []
    oof_prob = np.zeros((len(y_all), n_classes), dtype=np.float32)
    covered = np.zeros(len(y_all), dtype=bool)

    if cfg.cv and cfg.cv > 1:
        n_splits = int(min(cfg.cv, min_class))
        if n_splits < 2:
            raise RuntimeError(f"sub-{data.subject}: 每个类别至少需要 2 个试次才能做交叉验证")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        splits = [(tr, te) for tr, te in skf.split(X_all, y_all)]
    else:
        idx = np.arange(len(y_all))
        tr_idx, te_idx = train_test_split(
            idx, test_size=cfg.test_size, random_state=cfg.seed, stratify=y_all
        )
        splits = [(tr_idx, te_idx)]
        n_splits = 1

    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        # 从训练折里再切验证集用于早停
        if cfg.val_size and cfg.val_size > 0 and len(tr_idx) >= 10:
            tr2, val2 = train_test_split(
                tr_idx, test_size=cfg.val_size, random_state=cfg.seed + fold, stratify=y_all[tr_idx]
            )
        else:
            tr2, val2 = tr_idx, tr_idx

        X_tr, X_te = _normalize_split(X_all[tr2], y_all[tr2], X_all[te_idx], cfg.normalization)
        _, X_val = _normalize_split(X_tr, y_all[tr2], X_all[val2], cfg.normalization)

        if verbose:
            print(f"    折 {fold}/{n_splits}: 训练 {len(tr2)} / 验证 {len(val2)} / 测试 {len(te_idx)}")

        fit = fit_model(
            cfg, X_tr, y_all[tr2], X_val, y_all[val2], data.sfreq, device,
            seed=cfg.seed + 1000 * fold, verbose=verbose,
        )
        histories.append(fit.history)

        prob = predict_proba(fit.model, X_te, device)
        oof_prob[te_idx] = prob
        covered[te_idx] = True
        fold_acc = float((prob.argmax(axis=1) == y_all[te_idx]).mean())
        fold_accs.append(fold_acc)
        if verbose:
            print(f"      测试准确率 = {fold_acc:.4f} (最优 epoch {fit.best_epoch})")

        torch.save(
            {
                "model_state_dict": fit.model.state_dict(),
                "subject": int(data.subject),
                "fold": fold,
                "n_channels": int(X_tr.shape[1]),
                "n_times": int(X_tr.shape[2]),
                "sfreq": float(data.sfreq),
                "class_names": data.class_names,
                "config": cfg.to_dict(),
            },
            out_dir / (f"model_fold{fold}.pt" if n_splits > 1 else "model.pt"),
        )

    metrics = compute_metrics(y_all[covered], oof_prob[covered], data.class_names)
    metrics.update(
        {
            "subject": int(data.subject),
            "n_trials": int(len(y_all)),
            "n_channels": int(X_all.shape[1]),
            "n_times": int(X_all.shape[2]),
            "sfreq": float(data.sfreq),
            "cv_folds": int(n_splits),
            "fold_accuracies": [float(a) for a in fold_accs],
            "fold_accuracy_mean": float(np.mean(fold_accs)),
            "fold_accuracy_std": float(np.std(fold_accs)),
            "chance_level": 1.0 / n_classes,
            "elapsed_sec": float(time.time() - t0),
        }
    )

    # ---- 落盘 ----
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "history.json").write_text(
        json.dumps(histories, ensure_ascii=False), encoding="utf-8"
    )
    np.save(out_dir / "oof_prob.npy", oof_prob)
    np.save(out_dir / "labels.npy", y_all)
    if data.unresolved_channels:
        (out_dir / "unresolved_channels.txt").write_text(
            "\n".join(data.unresolved_channels), encoding="utf-8"
        )
    return metrics
