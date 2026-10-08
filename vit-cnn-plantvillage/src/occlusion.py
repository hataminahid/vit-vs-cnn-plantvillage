from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from src.masks import get_masks

GRAY_VALUE = 128


def occlude_pixels(pil_img: Image.Image, mask_bool: np.ndarray, fraction: float, rng: np.random.Generator):
    """Occludes `fraction` of the True pixels in mask_bool. Returns (image, n_occluded)."""
    img = np.array(pil_img.convert("RGB")).copy()
    idx = np.argwhere(mask_bool)
    if len(idx) == 0:
        return Image.fromarray(img), 0
    n_occ = int(len(idx) * fraction)
    if n_occ <= 0:
        return Image.fromarray(img), 0
    chosen = idx[rng.choice(len(idx), size=n_occ, replace=False)]
    img[chosen[:, 0], chosen[:, 1]] = GRAY_VALUE
    return Image.fromarray(img), n_occ


def occlude_n_pixels(pil_img: Image.Image, mask_bool: np.ndarray, n_pixels: int, rng: np.random.Generator):
    """Occludes exactly n_pixels random pixels from within mask_bool (area-matched control)."""
    img = np.array(pil_img.convert("RGB")).copy()
    idx = np.argwhere(mask_bool)
    if len(idx) == 0 or n_pixels <= 0:
        return Image.fromarray(img)
    n_pixels = min(n_pixels, len(idx))
    chosen = idx[rng.choice(len(idx), size=n_pixels, replace=False)]
    img[chosen[:, 0], chosen[:, 1]] = GRAY_VALUE
    return Image.fromarray(img)


def predict_single(model: nn.Module, transform, pil_img: Image.Image, device: torch.device) -> int:
    model.eval()
    x = transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1).item()
    return pred


def select_occlusion_indices(
    test_subset,
    to_label_idx: Callable,
    class_names: List[str],
    diseased_classes: List[str],
    n_per_class: int,
    seed: int,
) -> List[int]:
    by_class = {c: [] for c in diseased_classes}
    for i in range(len(test_subset)):
        ex = test_subset[i]
        cname = class_names[to_label_idx(ex["label"])]
        if cname in by_class:
            by_class[cname].append(i)

    rng = np.random.default_rng(seed)
    indices = []
    for idxs in by_class.values():
        idxs = np.array(idxs)
        if len(idxs) == 0:
            continue
        k = min(len(idxs), n_per_class)
        indices.extend(rng.choice(idxs, size=k, replace=False).tolist())
    return indices


def run_occlusion_experiment(
    test_subset,
    to_label_idx: Callable,
    occ_indices: List[int],
    models_and_tf: List[Tuple[str, nn.Module, Callable]],
    severities: List[float],
    min_disease_pixels: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    skipped = 0

    for idx in tqdm(occ_indices, desc="occlusion experiment"):
        ex = test_subset[int(idx)]
        img = ex["image"].convert("RGB")
        true_label = to_label_idx(ex["label"])

        leaf_mask, healthy_mask, disease_mask = get_masks(img)
        n_disease_total = int(disease_mask.sum())
        if n_disease_total < min_disease_pixels:
            skipped += 1
            continue

        for model_name, model, transform in models_and_tf:
            pred = predict_single(model, transform, img, next(model.parameters()).device)
            records.append(dict(image_idx=int(idx), true_label=true_label, model=model_name,
                                 condition="original", severity=0.0, pred=pred,
                                 correct=int(pred == true_label)))

        for level in severities:
            disease_img, n_occ = occlude_pixels(img, disease_mask, level, rng)
            random_img = occlude_n_pixels(img, leaf_mask, n_occ, rng)
            healthy_img = occlude_n_pixels(img, healthy_mask, n_occ, rng)

            for cond_name, cond_img in [("disease", disease_img), ("random", random_img), ("healthy", healthy_img)]:
                for model_name, model, transform in models_and_tf:
                    pred = predict_single(model, transform, cond_img, next(model.parameters()).device)
                    records.append(dict(image_idx=int(idx), true_label=true_label, model=model_name,
                                         condition=cond_name, severity=level, pred=pred,
                                         correct=int(pred == true_label)))

    print(f"Skipped (disease mask too small): {skipped}")
    return pd.DataFrame(records)


def summarize_occlusion(results_df: pd.DataFrame) -> pd.DataFrame:
    def macro_f1_group(g):
        return f1_score(g["true_label"], g["pred"], average="macro", zero_division=0)

    acc = results_df.groupby(["model", "condition", "severity"]).agg(
        accuracy=("correct", "mean"), n=("correct", "size")).reset_index()
    f1 = results_df.groupby(["model", "condition", "severity"]).apply(macro_f1_group).reset_index(name="macro_f1")
    summary = acc.merge(f1, on=["model", "condition", "severity"])
    return summary.sort_values(["model", "condition", "severity"]).reset_index(drop=True)
