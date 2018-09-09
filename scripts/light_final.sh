
#!/bin/bash

export PYTHONPATH="$(pwd)"

fixed_arc="1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 2 0 2 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 3 0 0 0 0 0 1 0 0 0 3 0 0 0 0 0 0 1 0 0 4 1 0 0 0 0 0 1 1 0 3 1 0 0 0 0 0 0 0 1 1 1"
fixed_arc="$fixed_arc 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 2 0 0 1 1 0 0 0 0 0 4 0 2 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 1 0 0 0 0 2 0 0 0 0 0 0 1 1 0 1 1 0 0 0 0 0 0 1 1 3 1"
# --data_format="NCHW" \
python src/main.py \
  --data_format="NHWC" \
  --search_for="micro" \
  --reset_output_dir \
  --data_path="data/cifar10" \
  --output_dir="outputs/out" \
  --summaries_dir="elog/log" \
  --batch_size=128 \
  --num_gpus=1 \
  --num_cpus=10 \
  --num_epochs=200 \
  --path_pool_size=1 \
  --log_every=50 \
  --eval_every=23 \
  --eval_every_epochs=1 \
  --child_fixed_arc="${fixed_arc}" \
  --child_use_aux_heads \
  --child_num_layers=30 \
  --child_out_filters=36 \
  --child_l2_reg=2e-4 \
  --child_num_branches=5 \
  --child_num_cells=8 \
  --child_keep_prob=0.80 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.0005 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --controller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=10 \
  --controller_train_steps=30 \
  --controller_lr=0.0035 \
  --controller_tanh_constant=1.10 \
  --controller_op_tanh_reduce=2.5 \
  "$@"

