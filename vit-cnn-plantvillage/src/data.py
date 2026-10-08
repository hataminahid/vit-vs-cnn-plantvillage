پimport io
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
import timm
import torch
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights

from src.config import CFG

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff")


def _find_column(column_names: List[str], candidates: List[str]) -> Optional[str]:
    """Case-insensitively finds the first candidate present in `column_names`.
    Returns None (instead of raising) if nothing matches, so callers can fall
    back to other detection strategies."""
    lower_map = {name.lower(): name for name in column_names}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _find_image_typed_column(features) -> Optional[str]:
    """Finds a column whose *feature type* is datasets.Image, regardless of name."""
    for name, feat in features.items():
        if feat.__class__.__name__ == "Image":
            return name
    return None


def _find_path_like_column(hf_split) -> Optional[str]:
    """Finds a plain string column whose values look like relative file paths to
    images (e.g. 'raw/color/Apple___Black_rot/xxxx.JPG'). This is the shape the
    mohanty/PlantVillage-derived (BrandonFors/Plant-Diseases-PlantVillage-Dataset) auto-converted
    parquet sometimes exposes as a single
    'text' column, instead of separate decoded 'image' + 'label' columns."""
    for name, feat in hf_split.features.items():
        if feat.__class__.__name__ != "Value" or getattr(feat, "dtype", "") != "string":
            continue
        sample = hf_split[0][name]
        if isinstance(sample, str) and "/" in sample and sample.lower().endswith(IMAGE_EXTENSIONS):
            return name
    return None


def _label_from_path(path: str) -> str:
    """Extracts the class name from a path like 'raw/color/Apple___Black_rot/xxxx.JPG'
    -> 'Apple___Black_rot' (the parent directory of the image file)."""
    parts = path.split("/")
    return parts[-2] if len(parts) >= 2 else "unknown"


def load_plantvillage():
    """Loads the PlantVillage dataset (BrandonFors/Plant-Diseases-PlantVillage-Dataset,
    a properly-schema'd mirror of mohanty/PlantVillage) from the Hugging Face Hub
    and returns (raw_train, raw_test, class_names, to_label_idx).

    Handles two possible schemas, since the auto-converted parquet version of
    this dataset has been observed to expose either:
      (a) proper decoded 'image' + 'label' columns, or
      (b) a single string column (often called 'text') holding relative file
          paths like 'raw/color/Apple___Black_rot/xxxx.JPG', with no decoded
          image or separate label column at all.
    """
    if CFG.get("DATASET_CONFIG"):
        raw = load_dataset(CFG["DATASET_NAME"], CFG["DATASET_CONFIG"])
    else:
        raw = load_dataset(CFG["DATASET_NAME"])
    raw_train, raw_test = raw["train"], raw["test"]

    print(f"[load_plantvillage] train columns: {raw_train.column_names}")
    print(f"[load_plantvillage] test columns:  {raw_test.column_names}")

    label_col = _find_column(
        raw_train.column_names,
        ["label", "labels", "class", "class_name", "category", "diagnosis"],
    )
    image_col = _find_column(
        raw_train.column_names,
        ["image", "img", "picture", "photo"],
    ) or _find_image_typed_column(raw_train.features)

    if label_col is None or image_col is None:
        path_col = _find_path_like_column(raw_train)
        if path_col is None:
            raise KeyError(
                f"Could not find usable label/image columns among "
                f"{raw_train.column_names!r}, and no path-like string column "
                f"was found either. Please inspect the printed column names "
                f"above and update src/data.py accordingly."
            )
        print(f"[load_plantvillage] no decoded image/label columns found; "
              f"falling back to path column {path_col!r} (label parsed from "
              f"the parent folder name, image downloaded lazily per-sample).")

        if path_col != "path":
            raw_train = raw_train.rename_column(path_col, "path")
            raw_test = raw_test.rename_column(path_col, "path")

        raw_train = raw_train.map(lambda ex: {"label": _label_from_path(ex["path"])})
        raw_test = raw_test.map(lambda ex: {"label": _label_from_path(ex["path"])})

        base_url = f"https://huggingface.co/datasets/{CFG['DATASET_NAME']}/resolve/main/"
        raw_train = raw_train.map(lambda ex: {"image_url": base_url + ex["path"]})
        raw_test = raw_test.map(lambda ex: {"image_url": base_url + ex["path"]})

        class_names = sorted(set(raw_train["label"]) | set(raw_test["label"]))
        name2idx = {n: i for i, n in enumerate(class_names)}

        def to_label_idx(v):
            return name2idx[v] if isinstance(v, str) else v

        return raw_train, raw_test, class_names, to_label_idx

    # Normalize column names to "label" / "image" so the rest of the module
    # (which is written against those fixed names) keeps working unchanged.
    if label_col != "label":
        raw_train = raw_train.rename_column(label_col, "label")
        raw_test = raw_test.rename_column(label_col, "label")
    if image_col != "image":
        raw_train = raw_train.rename_column(image_col, "image")
        raw_test = raw_test.rename_column(image_col, "image")

    label_feature = raw_train.features["label"]
    if hasattr(label_feature, "int2str"):
        class_names = list(label_feature.names)

        def to_label_idx(v):
            return v if isinstance(v, int) else class_names.index(v)
    else:
        class_names = sorted(set(raw_train["label"]) | set(raw_test["label"]))
        name2idx = {n: i for i, n in enumerate(class_names)}

        def to_label_idx(v):
            return name2idx[v] if isinstance(v, str) else v

    return raw_train, raw_test, class_names, to_label_idx


def stratified_indices(hf_split, num_classes: int, to_label_idx: Callable, max_per_class: int, seed: int) -> List[int]:
    """Samples up to `max_per_class` indices per class using only the label column
    (fast — does not decode any images)."""
    labels = hf_split["label"]
    by_class = {i: [] for i in range(num_classes)}
    for idx, lab in enumerate(labels):
        by_class[to_label_idx(lab)].append(idx)

    rng = np.random.default_rng(seed)
    chosen = []
    for idxs in by_class.values():
        idxs = np.array(idxs)
        if len(idxs) > max_per_class:
            idxs = rng.choice(idxs, size=max_per_class, replace=False)
        chosen.extend(idxs.tolist())
    rng.shuffle(chosen)
    return chosen


def build_vit(num_classes: int):
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    data_cfg = timm.data.resolve_data_config({}, model=model)
    img_size = data_cfg["input_size"][-1]
    mean = list(data_cfg["mean"])
    std = list(data_cfg["std"])
    return model, img_size, mean, std


def build_resnet(num_classes: int):
    import torch.nn as nn
    from torchvision.models import resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    img_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return model, img_size, mean, std


def make_transform(img_size: int, mean: List[float], std: List[float], train: bool):
    if train:
        ops = [T.RandomResizedCrop(img_size, scale=(0.7, 1.0)), T.RandomHorizontalFlip()]
    else:
        ops = [T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size)]
    ops += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    return T.Compose(ops)


class HFImageDataset(Dataset):
    """Wraps a Hugging Face `datasets` split/subset as a torch Dataset."""

    def __init__(self, hf_subset, to_label_idx: Callable, transform):
        self.data = hf_subset
        self.to_label_idx = to_label_idx
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        if "image" in ex and ex["image"] is not None:
            img = ex["image"]
        elif "image_url" in ex:
            # Fallback schema: dataset only exposed file paths, no decoded
            # image column. Download the raw bytes lazily, per sample.
            resp = requests.get(ex["image_url"], timeout=30)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
        else:
            raise KeyError(
                f"Example has neither 'image' nor 'image_url': {list(ex.keys())!r}"
            )
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = self.to_label_idx(ex["label"])
        return self.transform(img), label


def make_loaders(
    train_subset,
    test_subset,
    to_label_idx: Callable,
    img_size: int,
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_tf = make_transform(img_size, mean, std, train=True)
    eval_tf = make_transform(img_size, mean, std, train=False)
    train_ds = HFImageDataset(train_subset, to_label_idx, train_tf)
    test_ds = HFImageDataset(test_subset, to_label_idx, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, test_loader
