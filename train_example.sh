#!/usr/bin/env bash

set -eu
set -o pipefail

main() {
  echo 'preparing the dataset'
  python process_dataset.py
  echo 'starting training'
  python train.py --dataset_path example/processed_dataset/fonts.ds
}

main "$@"