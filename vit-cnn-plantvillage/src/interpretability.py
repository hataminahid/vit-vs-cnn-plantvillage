
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class AttentionRolloutExtractor:
    """Hooks attn_drop in every ViT block to capture the post-softmax attention matrix."""

    def __init__(self, model):
        self.model = model
        self.attentions = []
        self.hooks = []
        for blk in model.blocks:
            if hasattr(blk.attn, "fused_attn"):
                blk.attn.fused_attn = False  # force the eager path so attn is materialized
            h = blk.attn.attn_drop.register_forward_hook(self._hook)
            self.hooks.append(h)

    def _hook(self, module, inp, out):
        self.attentions.append(inp[0].detach().cpu())

    def clear(self):
        self.attentions = []

    def remove(self):
        for h in self.hooks:
            h.remove()


def compute_attention_rollout(attentions):
    """attentions: list of [1, heads, N, N] tensors, one per layer (batch size must be 1)."""
    n_tokens = attentions[0].size(-1)
    result = torch.eye(n_tokens).unsqueeze(0)
    for attn in attentions:
        attn_mean = attn.mean(dim=1)
        attn_mean = attn_mean + torch.eye(n_tokens).unsqueeze(0)
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
        result = torch.bmm(attn_mean, result)
    return result[0]  # [N, N]


def get_vit_attention_map(model, pil_img: Image.Image, transform, img_size: int, device):
    extractor = AttentionRolloutExtractor(model)
    model.eval()
    x = transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    rollout = compute_attention_rollout(extractor.attentions)
    extractor.remove()
    pred = logits.argmax(dim=1).item()

    cls_to_patches = rollout[0, 1:]
    grid = int(cls_to_patches.numel() ** 0.5)
    cam = cls_to_patches.reshape(grid, grid).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (img_size, img_size))
    return cam, pred


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def get_resnet_cam_map(model, pil_img: Image.Image, transform, img_size: int, device):
    model.eval()
    gradcam = GradCAM(model, model.layer4[-1])
    x = transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    x.requires_grad_(True)
    cam, pred = gradcam(x)
    return cam, pred


def plot_side_by_side(fig_path, samples, vit_model, resnet_model, vit_tf, resnet_tf,
                       vit_size, resnet_size, class_names, device):
    """samples: list of (pil_img, true_label_idx). Saves a comparison figure to fig_path."""
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row, (img, true_label) in enumerate(samples):
        vit_cam, vit_pred = get_vit_attention_map(vit_model, img, vit_tf, vit_size, device)
        rn_cam, rn_pred = get_resnet_cam_map(resnet_model, img, resnet_tf, resnet_size, device)

        img_vit = img.resize((vit_size, vit_size))
        img_rn = img.resize((resnet_size, resnet_size))

        axes[row, 0].imshow(img_vit)
        axes[row, 0].set_title(f"Original\ntrue={class_names[true_label]}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(img_vit)
        axes[row, 1].imshow(vit_cam, cmap="jet", alpha=0.5)
        axes[row, 1].set_title(f"ViT rollout\npred={class_names[vit_pred]}", fontsize=8)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(img_rn)
        axes[row, 2].imshow(rn_cam, cmap="jet", alpha=0.5)
        axes[row, 2].set_title(f"ResNet Grad-CAM\npred={class_names[rn_pred]}", fontsize=8)
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
