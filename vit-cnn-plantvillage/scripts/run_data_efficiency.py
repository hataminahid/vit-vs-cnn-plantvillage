
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CFG, SEED, get_device, set_seed
from src.data import build_resnet, build_vit, load_plantvillage, make_loaders, stratified_indices
from src.train import evaluate, train_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=CFG["DATA_EFFICIENCY_SIZES"],
                    help="training samples per class for each sweep point")
    p.add_argument("--epochs", type=int, default=CFG["DATA_EFFICIENCY_EPOCHS"],
                    help="matched epoch budget used at every sweep point")
    p.add_argument("--batch-size", type=int, default=CFG["BATCH_SIZE"])
    p.add_argument("--lr", type=float, default=CFG["LR"])
    p.add_argument("--max-test-per-class", type=int, default=CFG["MAX_TEST_PER_CLASS"])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print("Device:", device)
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)

    raw_train, raw_test, class_names, to_label_idx = load_plantvillage()
    num_classes = len(class_names)

    # Fixed test subset shared across every sweep point / both models, so accuracy
    # differences reflect training-set size only, not a moving evaluation target.
    test_idx = stratified_indices(raw_test, num_classes, to_label_idx, args.max_test_per_class, SEED + 1)
    test_subset = raw_test.select(test_idx)

    records = []
    for size in args.sizes:
        print(f"\n=== Data-efficiency sweep: {size} samples/class ===")
        train_idx = stratified_indices(raw_train, num_classes, to_label_idx, size, SEED)
        train_subset = raw_train.select(train_idx)
        print(f"train subset: {len(train_subset)}  test subset: {len(test_subset)}")

        # Fresh pretrained weights at every sweep point (no weight reuse across sizes),
        # so each point is an independent, fair fine-tune under the matched budget.
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

        train_model(vit_model, vit_train_loader, vit_test_loader, device,
                    name=f"ViT-B/16 (n={size})", epochs=args.epochs, lr=args.lr,
                    weight_decay=CFG["WEIGHT_DECAY"])
        train_model(resnet_model, resnet_train_loader, resnet_test_loader, device,
                    name=f"ResNet-50 (n={size})", epochs=args.epochs, lr=args.lr,
                    weight_decay=CFG["WEIGHT_DECAY"])

        vit_acc, vit_f1 = evaluate(vit_model, vit_test_loader, device)
        resnet_acc, resnet_f1 = evaluate(resnet_model, resnet_test_loader, device)
        print(f"n={size}: ViT acc={vit_acc:.4f} f1={vit_f1:.4f} | "
              f"ResNet acc={resnet_acc:.4f} f1={resnet_f1:.4f}")

        records.append(dict(model="ViT-B/16", train_per_class=size,
                             train_size=len(train_subset), accuracy=vit_acc, macro_f1=vit_f1))
        records.append(dict(model="ResNet-50", train_per_class=size,
                             train_size=len(train_subset), accuracy=resnet_acc, macro_f1=resnet_f1))

        # Free GPU memory between sweep points.
        del vit_model, resnet_model
        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(records)
    out_csv = os.path.join(CFG["RESULTS_DIR"], "data_efficiency_results.csv")
    results_df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, g in results_df.groupby("model"):
        g = g.sort_values("train_size")
        axes[0].plot(g["train_size"], g["accuracy"], marker="o", label=name)
        axes[1].plot(g["train_size"], g["macro_f1"], marker="o", label=name)
    axes[0].set_title("Accuracy vs. training-set size")
    axes[0].set_xlabel("training images (total, all classes)")
    axes[0].set_ylabel("test accuracy")
    axes[0].legend()
    axes[1].set_title("Macro-F1 vs. training-set size")
    axes[1].set_xlabel("training images (total, all classes)")
    axes[1].set_ylabel("test macro-F1")
    axes[1].legend()
    plt.tight_layout()
    out_png = os.path.join(CFG["RESULTS_DIR"], "data_efficiency_curve.png")
    plt.savefig(out_png, dpi=150)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
