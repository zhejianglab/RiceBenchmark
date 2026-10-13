#!/bin/bash
# Run the OneGenome-Rice harness on one Botanic1 size.
#
#   run_botanic1.sh SIZE LAYERS DATASET [DATASET ...]
#   run_botanic1.sh S 6,12,18,24 chromatin_access_512 smallRNA
#
# Environment:
#   HARNESS  OneGenome-Rice/evaluation/benchmark_code (commit 79984da2)
#   DATA     RiceBenchmark dataset snapshot (HF revision 1aeac87a)
#   WORK     output root; expects $WORK/models/Botanic1-$SIZE from make_model_dir.py
#   PYTHON   interpreter with the harness requirements and mamba_ssm
#   EXTRACT_SPLIT  rows per embedding part (default 50000, the harness default)
# Settings other than the paths, dataset list, layers, gpu_list and batch_size
# are the harness defaults from benchmark_code/config.yaml.
set -euo pipefail
SIZE=$1
LAYERS=$2
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
DATASETS=$(printf '  - %s\n' "$@")
TAG=$(printf '%s ' "$@" | md5sum | cut -c1-8)
CFG=$WORK/configs/Botanic1-$SIZE-$TAG.yaml
mkdir -p "$WORK/configs"
cat > "$CFG" <<EOF
model_path: "$WORK/models/Botanic1-$SIZE"
datasets_feature_path: "$HERE/datasets_info.yaml"
dataset_path: "$DATA"
eval_datasets:
$DATASETS
layer_to_eval: [$LAYERS]
gpu_list: [0]
seed: 42
embedding_output_dir: "$WORK/embeddings"
embedding_extract_split: ${EXTRACT_SPLIT:-50000}
batch_size: 32
eval_result_path: "$WORK/results"
classifer_type: "MLP"
pooled_embeddings_cat_dim: 0
process_power_per_gpu: 5
embedding_process_power: 4
eval_process_power: 1
mlp_dropout: 0.2
mlp_lr: 0.0001
mlp_epochs: 100
save_last_epoch_model: False
rf_n_estimators: 100
xgb_n_estimators: 100
xgb_eval_metric: 'mlogloss'
xgb_learning_rate: 0.1
xgb_max_depth: 6
target_best_table:
  - "accuracy"
  - "roc_auc"
  - ["roc_auc", "accuracy"]
  - "pearsonr"
target_layer_line_figure:
  - ["roc_auc", "accuracy"]
  - ["accuracy", "roc_auc"]
  - "pearsonr"
wandb_report: False
wandb_project: "mlp"
wandb_entity: "wandb_entity"
wandb_online: False
EOF
echo "config $CFG"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$HARNESS"
"$PYTHON" benchmarks.py --config "$CFG"
