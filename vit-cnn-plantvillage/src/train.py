from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(y.numpy().tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    return acc, f1


def get_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(y.numpy().tolist())
    return np.array(trues), np.array(preds)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    name: str,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
) -> pd.DataFrame:
    """Fine-tunes `model` under a fixed (matched) training budget: same optimizer type,
    schedule, epoch count and batch size are expected to be passed identically for both
    architectures by the caller."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"[{name}] epoch {epoch + 1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=running_loss / max(1, n_seen))
        train_loss = running_loss / max(1, n_seen)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"[{name}] epoch {epoch + 1}: train_loss={train_loss:.4f} "
              f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")
        history.append(dict(model=name, epoch=epoch + 1, train_loss=train_loss,
                             val_acc=val_acc, val_macro_f1=val_f1))
    return pd.DataFrame(history)


def per_class_disagreement(vit_true, vit_pred, resnet_true, resnet_pred, class_names) -> pd.DataFrame:
    assert np.array_equal(vit_true, resnet_true), "loaders must share the same (non-shuffled) order"
    df = pd.DataFrame({
        "true": vit_true,
        "true_name": [class_names[t] for t in vit_true],
        "vit_pred": vit_pred,
        "resnet_pred": resnet_pred,
    })
    df["vit_correct"] = df["true"] == df["vit_pred"]
    df["resnet_correct"] = df["true"] == df["resnet_pred"]

    vit_only = df[df.vit_correct & ~df.resnet_correct]
    resnet_only = df[~df.vit_correct & df.resnet_correct]

    summary = pd.DataFrame({
        "class": class_names,
        "vit_only_correct": [(vit_only.true_name == c).sum() for c in class_names],
        "resnet_only_correct": [(resnet_only.true_name == c).sum() for c in class_names],
    }).sort_values("vit_only_correct", ascending=False)
    return summary
