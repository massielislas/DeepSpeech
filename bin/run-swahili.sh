#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "data/swahili/swahili_training.csv" ]; then
    echo "Downloading and preprocessing swahili example data, saving in ./data/swahili."
    python -u bin/import_swahili.py ./data/swahili
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/swahili"))')
fi

# # Force only one visible device because we have a single-sample dataset
# # and when trying to run on multiple devices (like GPUs), this will break
# export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar \
  --train_files data/swahili/swahili_training.csv \
  --test_files data/swahili/swahili_training.csv \
  --train_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 100 \
  --epochs 200 \
  --checkpoint_dir "$checkpoint_dir" \
  "$@"
