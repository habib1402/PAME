# Datasets

PAME uses five public panoramic radiograph datasets. **We do not redistribute any image or annotation.**
Download each dataset from its source under its own license, then run the preparation scripts. The scripts rebuild
exactly the files used in the paper and check them against the lists in [`../splits/`](../splits).

| Dataset | Role | Images used | Source | License / terms |
|---|---|---|---|---|
| Tufts Dental Database | training | 1000 (700 / 100 / 200, seed 0) | https://tdd.ece.tufts.edu (request form) | research use under the Tufts terms |
| O2PR (OdontoAI public set) | training | 1996 (official 1696 / 300) | OdontoAI, Silva et al., CMBBE: Imaging & Visualization, 2023 | terms of the OdontoAI authors |
| AKUDENTAL | training | 333 (official fold 0) | https://github.com/melihoz/akudental | CC BY-NC-SA 4.0 |
| AKU OPG | unseen test | 123 of 250 (see below) | https://doi.org/10.5281/zenodo.10538750 | CC BY 4.0 |
| Dual-labeled | unseen test | 496 of the 500 released | https://www.kaggle.com/datasets/zwbzwb12341234/a-dual-labeled-dataset | CC BY-NC-ND 4.0 |

## Cleaning applied by `tools/prepare_data.py`

- **AKU OPG:** folder 3 (119 images) contains Tufts radiographs and 8 further images belong to UFBA-UESC; all 127 are
  removed (listed in [`../splits/removed_duplicates.txt`](../splits/removed_duplicates.txt)).
- **Dual-labeled:** 4 near-duplicate images removed.
- **AKUDENTAL:** the release mirrors the lower-jaw quadrants (quadrant 3 on the image left); quadrants 3 and 4 are swapped back.
- All datasets are mapped to 52 FDI classes (32 permanent, 20 primary); restorations, implants, and supernumerary labels are dropped.

## Expected layout

```
data/
  raw/
    tufts/            Radiographs/, Segmentation/teeth_polygon.json
    o2pr/             train/, valid/  (COCO export, each with _annotations.coco.json)
    akudental/        images/, annotations/, folds/fold_0/labels/{train,val,test}/
    aku_opg/          dataset/folder 1, folder 2, folder 3 (each with OPGs/ and annotations)
    dual_labeled/     images1/, labels/
  tufts_seed0/        created by tools/prepare_tufts.sh
  annotations/        created by tools/prepare_data.py
```

## Steps

```bash
# 1) Tufts: official SADP-Net conversion with a fixed seed (clone https://github.com/qklee-lz/SADP-Net first)
bash tools/prepare_tufts.sh /path/to/SADP-Net data/raw/tufts data 0
# 2) unified annotations for all datasets (prints "matches splits/" for every file)
python tools/prepare_data.py --tufts data/tufts_seed0 --o2pr data/raw/o2pr --akudental data/raw/akudental \
    --aku-opg data/raw/aku_opg --dual data/raw/dual_labeled --out data/annotations
```

The training configs identify the source of each image from its path (`AddSourceId` with the patterns
`Tufts`, `O2PR`, `AKUDENTAL`, case-insensitive), so keep these names in the folder names and avoid them elsewhere in
the path.
