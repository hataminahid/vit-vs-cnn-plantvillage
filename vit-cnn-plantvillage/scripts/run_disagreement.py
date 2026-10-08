
import os

from src.checkpoint import load_resnet, load_vit
from src.config import CFG, SEED, get_device, set_seed
from src.data import load_plantvillage, make_transform, stratified_indices, HFImageDataset
from src.train import get_predictions, per_class_disagreement
from torch.utils.data import DataLoader


def main():
    set_seed(SEED)
    device = get_device()
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)

    raw_train, raw_test, class_names, to_label_idx = load_plantvillage()
    num_classes = len(class_names)

    test_idx = stratified_indices(raw_test, num_classes, to_label_idx, CFG["MAX_TEST_PER_CLASS"], SEED + 1)
    test_subset = raw_test.select(test_idx)

    vit_model, vit_tf, _ = load_vit(device, num_classes)
    resnet_model, resnet_tf, _ = load_resnet(device, num_classes)

    vit_loader = DataLoader(HFImageDataset(test_subset, to_label_idx, vit_tf),
                             batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=CFG["NUM_WORKERS"])
    resnet_loader = DataLoader(HFImageDataset(test_subset, to_label_idx, resnet_tf),
                                batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=CFG["NUM_WORKERS"])

    vit_true, vit_pred = get_predictions(vit_model, vit_loader, device)
    resnet_true, resnet_pred = get_predictions(resnet_model, resnet_loader, device)

    summary = per_class_disagreement(vit_true, vit_pred, resnet_true, resnet_pred, class_names)
    out_path = os.path.join(CFG["RESULTS_DIR"], "per_class_disagreement.csv")
    summary.to_csv(out_path, index=False)
    print(summary.head(15))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
