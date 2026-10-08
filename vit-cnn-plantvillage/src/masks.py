"""Heuristic, colour-based approximation of the diseased-tissue region in a leaf image.

PlantVillage does not ship pixel-level disease annotations, so this module derives an
approximate mask from HSV colour statistics: the (mostly uniform, low-saturation) studio
background is separated from the leaf, and within the leaf, green healthy tissue is
separated from non-green (brown/yellow/dark) symptomatic tissue.

This is an approximation, not ground truth — see the README "Limitations" section.
"""
import cv2
import numpy as np
from PIL import Image


def get_masks(pil_img: Image.Image):
    """Returns (leaf_mask, healthy_mask, disease_mask) as boolean numpy arrays,
    same H×W as the input image."""
    img = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, _v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    bg_mask = s < 25  # near-white/near-gray studio background
    leaf_mask = ~bg_mask

    healthy_mask = (h >= 30) & (h <= 95) & (s > 40) & leaf_mask  # green tissue
    disease_mask = leaf_mask & ~healthy_mask  # remaining leaf area = spots/symptoms

    return leaf_mask, healthy_mask, disease_mask
