#!/usr/bin/env bash
# Tufts Dental Database -> COCO splits, exactly as in the paper: the official SADP-Net conversion
# (random 70/10/20 split, contour closing + Gaussian smoothing, Universal -> FDI numbering) with two changes:
#   1) the split is seeded (random.seed(SEED) on a sorted file list), so it is reproducible;
#   2) contour processing and COCO conversion run for train, val, and test.
# The SADP-Net scripts are NOT redistributed here; clone https://github.com/qklee-lz/SADP-Net yourself.
#
# Usage: bash tools/prepare_tufts.sh SADP_NET_REPO TUFTS_DIR OUT_DIR SEED [SEED ...]
#   TUFTS_DIR: the Tufts download, containing Radiographs/ and Segmentation/teeth_polygon.json
#   Output:    OUT_DIR/tufts_seed{SEED}/{train,val,test,annotations/instances_{train,val,test}.json}
# The paper uses seed 0 for multi-dataset training (splits/train_TOA.txt lists the resulting Tufts images).
set -euo pipefail
SADP=$(cd "$1" && pwd); SRC=$(cd "$2" && pwd); OUTROOT=$(mkdir -p "$3" && cd "$3" && pwd); shift 3
PY=${PY:-python}
OFFICIAL="$SADP/Tufts/Tufts2COCO"
[ -f "$OFFICIAL/spilit.py" ] || { echo "SADP-Net Tufts scripts not found in $OFFICIAL"; exit 1; }
for SEED in "$@"; do
  OUT="$OUTROOT/tufts_seed$SEED"; echo "=== seed $SEED -> $OUT"
  rm -rf "$OUT"; mkdir -p "$OUT/Tufts2COCO"
  ln -s "$SRC/Radiographs" "$OUT/Radiographs"
  ln -s "$SRC/Segmentation/teeth_polygon.json" "$OUT/teeth_polygon.json"
  cd "$OUT/Tufts2COCO"
  sed -e "s/^import shutil\r\?$/import shutil\nrandom.seed($SEED)/" \
      -e "s/images = \[f for f in os.listdir(data_dir)/images = [f for f in sorted(os.listdir(data_dir))/" \
      "$OFFICIAL/spilit.py" > spilit.py
  grep -q "random.seed($SEED)" spilit.py
  "$PY" spilit.py
  for S in train val test; do
    sed -e "s#instances_train_source#instances_${S}_source#" -e "s#image_dir = '../train'#image_dir = '../$S'#" \
        -e "s#instances_train_smooth#instances_${S}_smooth#" "$OFFICIAL/annotationProcess.py" > "annotationProcess_$S.py"
    sed -e "s#instances_train_smooth#instances_${S}_smooth#" -e "s#instances_train.json#instances_${S}.json#" \
        "$OFFICIAL/transformCOCO.py" > "transformCOCO_$S.py"
    "$PY" "annotationProcess_$S.py"; "$PY" "transformCOCO_$S.py"
  done
  cd - > /dev/null
done
