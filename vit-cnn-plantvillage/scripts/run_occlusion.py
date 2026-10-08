
import os

import matplotlib.pyplot as plt
import numpy as np

from src.checkpoint import load_resnet, load_vit
from src.config import CFG, SEED, get_device, set_seed
from src.data import load_plantvillage, stratified_indices
from src.occlusion import run_occlusion_experiment, select_occlusion_indices, summarize_occlusion


def main():
    set_seed(SEED)
    device = get_device()
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)

    raw_train, raw_test, class_names, to_label_idx = load_plantvillage()
    num_classes = len(class_names)
    diseased_classes = [c for c in class_names if "healthy" not in c.lower()]

    test_idx = stratified_indices(raw_test, num_classes, to_label_idx, CFG["MAX_TEST_PER_CLASS"], SEED + 1)
    test_subset = raw_test.select(test_idx)

    vit_model, vit_tf, _ = load_vit(device, num_classes)
    resnet_model, resnet_tf, _ = load_resnet(device, num_classes)
    models_and_tf = [("ViT", vit_model, vit_tf), ("ResNet", resnet_model, resnet_tf)]

    occ_indices = select_occlusion_indices(
        test_subset, to_label_idx, class_names, diseased_classes,
        n_per_class=CFG["N_PER_CLASS_OCC"], seed=SEED + 3)
    print(f"Images selected for occlusion experiment: {len(occ_indices)}")

    results_df = run_occlusion_experiment(
        test_subset, to_label_idx, occ_indices, models_and_tf,
        severities=CFG["SEVERITIES"], min_disease_pixels=CFG["MIN_DISEASE_PIXELS"], seed=SEED + 3)
    results_df.to_csv(os.path.join(CFG["RESULTS_DIR"], "occlusion_results_raw.csv"), index=False)

    summary = summarize_occlusion(results_df)
    summary.to_csv(os.path.join(CFG["RESULTS_DIR"], "occlusion_summary.csv"), index=False)
    print(summary)

    baseline_acc = summary[summary.condition == "original"].set_index("model")["accuracy"].to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, model_name in zip(axes, ["ViT", "ResNet"]):
        sub = summary[(summary.model == model_name) & (summary.condition != "original")]
        for cond, g in sub.groupby("condition"):
            g = g.sort_values("severity")
            ax.plot(g["severity"], g["accuracy"], marker="o", label=cond)
        ax.axhline(baseline_acc.get(model_name, np.nan), color="gray", linestyle="--", label="original")
        ax.set_title(model_name)
        ax.set_xlabel("severity (fraction of disease-region pixels occluded)")
        ax.set_ylabel("accuracy")
        ax.legend()
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["RESULTS_DIR"], "occlusion_accuracy_curves.png"), dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for model_name in ["ViT", "ResNet"]:
        base = baseline_acc.get(model_name, np.nan)
        g = summary[(summary.model == model_name) & (summary.condition == "disease")].sort_values("severity")
        drop = base - g["accuracy"]
        ax2.plot(g["severity"], drop, marker="o", label=model_name)
    ax2.set_xlabel("severity (fraction of disease-region pixels occluded)")
    ax2.set_ylabel("accuracy drop vs. original")
    ax2.set_title("Accuracy drop when occluding the disease region")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["RESULTS_DIR"], "accuracy_drop_disease_condition.png"), dpi=150)

    print("Saved results to", CFG["RESULTS_DIR"])


if __name__ == "__main__":
    main()
