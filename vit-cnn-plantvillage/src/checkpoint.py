import os

import torch

from src.config import CFG
from src.data import build_resnet, build_vit, make_transform


def load_vit(device, num_classes: int, checkpoint_path: str = None):
    checkpoint_path = checkpoint_path or os.path.join(CFG["CHECKPOINT_DIR"], "vit.pt")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model, img_size, mean, std = build_vit(ckpt.get("num_classes", num_classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    img_size, mean, std = ckpt["img_size"], ckpt["mean"], ckpt["std"]
    eval_tf = make_transform(img_size, mean, std, train=False)
    return model, eval_tf, img_size


def load_resnet(device, num_classes: int, checkpoint_path: str = None):
    checkpoint_path = checkpoint_path or os.path.join(CFG["CHECKPOINT_DIR"], "resnet.pt")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model, img_size, mean, std = build_resnet(ckpt.get("num_classes", num_classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    img_size, mean, std = ckpt["img_size"], ckpt["mean"], ckpt["std"]
    eval_tf = make_transform(img_size, mean, std, train=False)
    return model, eval_tf, img_size
