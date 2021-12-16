#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2
bigbird/classifier/run_classifier.py \
  --data_dir="tfds://imdb_reviews/plain_text" \
  --output_dir="$GCP_EXP_BUCKET"classifier/imdb \
  --attention_type=block_sparse \
  --max_encoder_length=4096 \
  --num_attention_heads=16 \
  --num_hidden_layers=24 \
  --hidden_size=1024 \
  --intermediate_size=4096 \
  --block_size=64 \
  --train_batch_size=1 \
  --eval_batch_size=1 \
  --do_train=True \
  --do_eval=True \
  --use_tpu=False \
  --init_checkpoint=ckpt/pretrain/bigbr_large/model.ckpt-0
