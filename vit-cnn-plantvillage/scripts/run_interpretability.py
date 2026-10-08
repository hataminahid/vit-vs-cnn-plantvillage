
import argparse
import os

import numpy as np

from src.checkpoint import load_resnet, load_vit
from src.config import CFG, SEED, get_device, set_seed
from src.data import load_plantvillage, stratified_indices
from src.interpretability import plot_side_by_side


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-examples", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)

    raw_train, raw_test, class_names, to_label_idx = load_plantvillage()
    num_classes = len(class_names)

    test_idx = stratified_indices(raw_test, num_classes, to_label_idx, CFG["MAX_TEST_PER_CLASS"], SEED + 1)
    test_subset = raw_test.select(test_idx)

    vit_model, vit_tf, vit_size = load_vit(device, num_classes)
    resnet_model, resnet_tf, resnet_size = load_resnet(device, num_classes)

    rng = np.random.default_rng(SEED + 2)
    sample_ids = rng.choice(len(test_subset), size=args.n_examples, replace=False)
    samples = []
    for idx in sample_ids:
        ex = test_subset[int(idx)]
        samples.append((ex["image"].convert("RGB"), to_label_idx(ex["label"])))

    out_path = os.path.join(CFG["RESULTS_DIR"], "attention_gradcam_comparison.png")
    plot_side_by_side(out_path, samples, vit_model, resnet_model, vit_tf, resnet_tf,
                       vit_size, resnet_size, class_names, device)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
