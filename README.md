# ViT-B/16 vs ResNet-50 on PlantVillage Targeted Symptom-Occlusion Study
Original contribution
Instead of applying random corruption to the whole image, this project runs a targeted occlusion experiment: a heuristic mask approximates the diseased-tissue region of each leaf, and we occlude a controlled fraction of that region (25% / 50% / 75%) to see how much of the disease signature each model still needs to classify correctly. Two controls are included so the comparison is fair:


Condition	What gets occluded	Purpose
original	nothing	baseline accuracy
random	same number of pixels, random location on the leaf	fair random baseline
disease	same number of pixels, sampled from the heuristic disease mask	main probe
healthy	same number of pixels, sampled from healthy leaf tissue	second control
Hypothesis
As more of the disease region is occluded, accuracy drops for both models. Because ViT-B/16 uses global self-attention (it can relate distant patches of the leaf to each other), we expect it to degrade more slowly than ResNet-50 — which relies on local receptive fields — when only part of the symptom is masked, especially at the lower severities (25% / 50%).

Honest limitation
PlantVillage does not provide pixel-level disease annotations. The “disease mask” here is a colour heuristic in HSV space (green tissue = healthy, non-green tissue inside the leaf = symptomatic), not a ground-truth segmentation. It will misclassify some healthy dark tissue or leaf veins as “disease” and vice versa. This is stated explicitly so it can be discussed in the report’s Limitations section see src/masks.py.

Dataset
BrandonFors/Plant-Diseases-PlantVillage-Dataset on the Hugging Face Hub a properly-schema’d (image + label columns) mirror of the Mohanty et al. (2016) PlantVillage dataset, 54,305 leaf images, 38 crop-disease classes (incl. healthy), with a pre-built train/test split (43.5k / 10.8k). No manual download needed — datasets.load_dataset fetches and caches it automatically the first time a script runs.

Note: the original mohanty/PlantVillage repo on the Hub currently auto-converts to a single text column of file paths (no decoded image/label columns), which is why this mirror is used instead. src/data.py still contains a defensive fallback path in case a future dataset swap runs into the same issue again.

bibtex
@article{Mohanty_Hughes_Salathe_2016,
  title   = {Using deep learning for image-based plant disease detection},
  volume  = {7},
  DOI     = {10.3389/fpls.2016.01419},
  journal = {Frontiers in Plant Science},
  author  = {Mohanty, Sharada P. and Hughes, David P. and Salath\'e, Marcel},
  year    = {2016}
}
Repository layout
text
vit-cnn-plantvillage/
├── src/
│   ├── config.py            # CFG dict, fixed SEED=42, device helper
│   ├── data.py               # dataset loading, stratified sampling, transforms, model builders
│   ├── masks.py               # heuristic disease/healthy/leaf mask (HSV)
│   ├── occlusion.py          # occlusion primitives + the main experiment
│   ├── interpretability.py   # attention rollout (ViT) + Grad-CAM (ResNet)
│   ├── train.py               # matched-budget training loop, evaluation, disagreement analysis
│   └── checkpoint.py         # load saved checkpoints
├── scripts/
│   ├── train_models.py        # Step 1: fine-tune both models, save checkpoints
│   ├── run_disagreement.py    # Step 2: per-class ViT-vs-ResNet disagreement analysis
│   ├── run_interpretability.py# Step 3: attention rollout vs. Grad-CAM comparison figure
│   ├── run_occlusion.py       # Step 4: the main targeted-occlusion experiment
│   └── run_data_efficiency.py # Step 5 (stretch, optional): accuracy vs. training-set size
├── notebooks/
│   └── colab_quickstart.ipynb # clones this repo and runs all 4 steps in Colab
├── checkpoints/                # saved model weights (git-ignored, produced by Step 1)
├── results/                    # csv/png outputs (git-ignored, produced by Steps 1 -4)
├── requirements.txt
└── README.md
Reproducing the main result end-to-end
Random seeds are fixed (SEED = 42 in src/config.py, applied to random, numpy, and torch) so results are reproducible run-to-run on the same hardware.

Option A Google Colab (recommended, free GPU)
Push this repository to your own GitHub account (see below), or open notebooks/colab_quickstart.ipynb directly and edit the REPO_URL cell.
Runtime Change runtime type T4 GPU.
Run all cells. It clones the repo, installs dependencies (skipping torch/torchvision, which Colab already ships with a matching CUDA build), and runs all four steps in order.
Option B local machine / your own GPU server
bash
git clone https://github.com/<your-username>/vit-cnn-plantvillage.git
cd vit-cnn-plantvillage
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Step 1  fine-tune both models under a matched budget
python -m scripts.train_models --epochs 3 --max-train-per-class 150 --max-test-per-class 30

# Step 2  per-class disagreement analysis
python -m scripts.run_disagreement

# Step 3  attention rollout (ViT) vs Grad-CAM (ResNet) figure
python -m scripts.run_interpretability --n-examples 4

# Step 4  the main targeted symptom-occlusion experiment
python -m scripts.run_occlusion

# Step 5 (stretch, optional)  data-efficiency sweep: accuracy vs. training-set size
python -m scripts.run_data_efficiency --sizes 20 50 100 150 --epochs 2
All outputs (checkpoints, csv tables, png figures) land in checkpoints/ and results/.

Config knobs
Everything tunable lives in src/config.py (CFG dict) and can also be overridden via CLI flags on scripts/train_models.py. Lower MAX_TRAIN_PER_CLASS / MAX_TEST_PER_CLASS for a fast smoke test (e.g. 20/10) before committing to a full run.

Publishing this repo to GitHub
bash
cd vit-cnn-plantvillage
git init
git add .
git commit -m "Initial commit: ViT vs ResNet targeted symptom-occlusion study"
git branch -M main
git remote add origin https://github.com/<your-username>/vit-cnn-plantvillage.git
git push -u origin main
Report outputs to cite
results/training_history.csv, results/training_curves.png matched-budget training curves
results/per_class_disagreement.csv per-class ViT-only-correct vs ResNet-only-correct counts
results/attention_gradcam_comparison.png side-by-side interpretability figure
results/occlusion_results_raw.csv every individual prediction from the occlusion experiment
results/occlusion_summary.csv accuracy / macro-F1 per model * condition *severity
results/occlusion_accuracy_curves.png, results/accuracy_drop_disease_condition.png main figures for the report
results/data_efficiency_results.csv, results/data_efficiency_curve.png stretch bonus: accuracy/macro-F1 vs. training-set size for both models (produced by Step 5, optional)
