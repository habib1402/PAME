# Model zoo

All models: standard Mask2Former, Swin-T backbone (ImageNet-1k), 1024 x 512 input, 50 epochs, 52 FDI classes.
Mask mAP is class-aware COCO mask AP over the FDI classes present in each test set. Checkpoints contain weights only
(about 190 MB each).

**Download links will be added on publication** (hosting: GitHub Releases / Zenodo).

## Main models (Tufts + O2PR + AKUDENTAL)

| Model | Seed | Tufts | O2PR | AKUDENTAL | AKU-OPG* | Dual-labeled* | Checkpoint |
|---|---|---|---|---|---|---|---|
| Naive merge | 0, 1, 2 | 63.7 | 77.0 | 68.6 | 69.0 | 68.7 | coming soon |
| PAME, own layer / blend K = 3 | 0, 1, 2 | 63.6 | 76.4 | 69.0 | 76.4 | 76.0 | coming soon |

\* never used for training. Values are the mean over 3 seeds.

## Leave-one-dataset-out

| Model | Training data | Unseen | Checkpoint |
|---|---|---|---|
| Naive / PAME | Tufts + O2PR | AKUDENTAL | coming soon |
| Naive / PAME | Tufts + AKUDENTAL | O2PR | coming soon |
| PAME, O2PR layer fine-tuned (K = 3, 10; 3 draws each) | Tufts + AKUDENTAL + K O2PR images | O2PR | coming soon |

## Test-set predictions

COCO results files (RLE masks, FDI number, score, box) for every test set, produced by `tools/export_predictions.py`.
They contain no images and can be scored with `tools/evaluate.py` after building the annotations.

## Terms of use

Code is Apache 2.0. The weights and predictions are derived from datasets with non-commercial terms (Tufts, O2PR,
AKUDENTAL) and are released under **CC BY-NC-SA 4.0** for non-commercial research use only. They are not a medical
device and must not be used for clinical decisions.
