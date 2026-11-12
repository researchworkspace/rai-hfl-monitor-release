#!/usr/bin/env bash
# Download the dermatology data (Fed-ISIC2019) and compute its embeddings. Run from the repository root.
#
#   bash scripts/get_data.sh            download into data/isic/raw/ (about 9.8 GB), then embed
#   DEVICE=cuda bash scripts/get_data.sh
#
# The census data (ACS 2018 one-year) needs no step here: folktables downloads it into data/acs/ on first use.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
RAW=data/isic/raw
mkdir -p "$RAW"

# Centre assignment and train/test split of FLamby's Fed-ISIC2019.
[[ -f $RAW/flamby_train_test_split.csv ]] || curl -fL -o "$RAW/flamby_train_test_split.csv" \
  https://raw.githubusercontent.com/owkin/FLamby/main/flamby/datasets/fed_isic2019/dataset_creation_scripts/train_test_split
# ISIC 2019 challenge images and metadata (age and sex).
[[ -f $RAW/ISIC_2019_Training_Metadata.csv ]] || curl -fL -o "$RAW/ISIC_2019_Training_Metadata.csv" \
  https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv
[[ -f $RAW/ISIC_2019_Training_Input.zip ]] || curl -fL -o "$RAW/ISIC_2019_Training_Input.zip" \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip

# Frozen ImageNet EfficientNet-B0 embeddings of every image under 12 views (blur levels x noise).
[[ -f data/isic/embeddings.npz ]] || "$PY" -I scripts/raimon_cli.py isic-embed --raw "$RAW" \
  --out data/isic/embeddings.npz ${DEVICE:+--device $DEVICE}
