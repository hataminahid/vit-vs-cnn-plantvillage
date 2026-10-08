
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.config import CFG, get_device, set_seed, SEED
from src.data import build_resnet, build_vit, load_plantvillage, make_loaders, stratified_indices
from src.train import evaluate, train_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=CFG["EPOCHS"])
    p.add_argument("--batch-size", type=int, default=CFG["BATCH_SIZE"])
    p.add_argument("--lr", type=float, default=CFG["LR"])
    p.add_argument("--max-train-per-class", type=int, default=CFG["MAX_TRAIN_PER_CLASS"])
    p.add_argument("--max-test-per-class", type=int, default=CFG["MAX_TEST_PER_CLASS"])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print("Device:", device)

    os.makedirs(CFG["CHECKPOINT_DIR"], exist_ok=True)
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)

    raw_train, raw_test, class_names, to_label_idx = load_plantvillage()
    num_classes = len(class_names)
    print("num classes:", num_classes)

    with open(os.path.join(CFG["RESULTS_DIR"], "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    train_idx = stratified_indices(raw_train, num_classes, to_label_idx, args.max_train_per_class, SEED)
    test_idx = stratified_indices(raw_test, num_classes, to_label_idx, args.max_test_per_class, SEED + 1)
    train_subset = raw_train.select(train_idx)
    test_subset = raw_test.select(test_idx)
    print("train subset:", len(train_subset), " test subset:", len(test_subset))

    vit_model, vit_size, vit_mean, vit_std = build_vit(num_classes)
    resnet_model, resnet_size, resnet_mean, resnet_std = build_resnet(num_classes)
    vit_model.to(device)
    resnet_model.to(device)

    vit_train_loader, vit_test_loader = make_loaders(
        train_subset, test_subset, to_label_idx, vit_size, vit_mean, vit_std,
        args.batch_size, CFG["NUM_WORKERS"])
    resnet_train_loader, resnet_test_loader = make_loaders(
        train_subset, test_subset, to_label_idx, resnet_size, resnet_mean, resnet_std,
        args.batch_size, CFG["NUM_WORKERS"])

    vit_history = train_model(vit_model, vit_train_loader, vit_test_loader, device,
                               name="ViT-B/16", epochs=args.epochs, lr=args.lr,
                               weight_decay=CFG["WEIGHT_DECAY"])
    resnet_history = train_model(resnet_model, resnet_train_loader, resnet_test_loader, device,
                                  name="ResNet-50", epochs=args.epochs, lr=args.lr,
                                  weight_decay=CFG["WEIGHT_DECAY"])

    history_df = pd.concat([vit_history, resnet_history], ignore_index=True)
    history_df.to_csv(os.path.join(CFG["RESULTS_DIR"], "training_history.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, g in history_df.groupby("model"):
        axes[0].plot(g["epoch"], g["val_acc"], marker="o", label=name)
        axes[1].plot(g["epoch"], g["val_macro_f1"], marker="o", label=name)
    axes[0].set_title("Validation Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].set_title("Validation Macro-F1"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["RESULTS_DIR"], "training_curves.png"), dpi=150)

    final_vit_acc, final_vit_f1 = evaluate(vit_model, vit_test_loader, device)
    final_resnet_acc, final_resnet_f1 = evaluate(resnet_model, resnet_test_loader, device)
    print(f"Final ViT-B/16:  acc={final_vit_acc:.4f} macro_f1={final_vit_f1:.4f}")
    print(f"Final ResNet-50: acc={final_resnet_acc:.4f} macro_f1={final_resnet_f1:.4f}")

    torch.save({
        "state_dict": vit_model.state_dict(),
        "img_size": vit_size, "mean": vit_mean, "std": vit_std,
        "num_classes": num_classes,
    }, os.path.join(CFG["CHECKPOINT_DIR"], "vit.pt"))

    torch.save({
        "state_dict": resnet_model.state_dict(),
        "img_size": resnet_size, "mean": resnet_mean, "std": resnet_std,
        "num_classes": num_classes,
    }, os.path.join(CFG["CHECKPOINT_DIR"], "resnet.pt"))

    print("Saved checkpoints to", CFG["CHECKPOINT_DIR"])
    print("Saved training history/plots to", CFG["RESULTS_DIR"])


if __name__ == "__main__":
    main()
