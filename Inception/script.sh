#!/usr/bin/env bash

## Load Flower data
#DATA_DIR=/media/dataHD3/kpugdeet/inceptionModel/data/flowers
#python ./slim/download_and_convert_data.py --dataset_name=flowers --dataset_dir="${DATA_DIR}"

## Fine-tuning a model from an existing checkpoint
#DATASET_DIR=/media/dataHD3/kpugdeet/inceptionModel/data/flowers
#TRAIN_DIR=/media/dataHD3/kpugdeet/inceptionModel/flowers-models/inception_v4
#CHECKPOINT_PATH=/media/dataHD3/kpugdeet/inceptionModel/inception_v4.ckpt
#CUDA_VISIBLE_DEVICES=0 python ./slim/train_image_classifier.py \
#    --train_dir=${TRAIN_DIR} \
#    --dataset_dir=${DATASET_DIR} \
#    --dataset_name=flowers \
#    --dataset_split_name=train \
#    --model_name=inception_v4 \
#    --checkpoint_path=${CHECKPOINT_PATH} \
#    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
#    --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
#    --max_number_of_steps=1000 \
#    --batch_size=32 \
#    --learning_rate=0.01 \
#    --learning_rate_decay_type=fixed \
#    --save_interval_secs=60 \
#    --save_summaries_secs=60 \
#    --log_every_n_steps=100 \
#    --optimizer=rmsprop \
#    --weight_decay=0.00004

## Evaluating performance of a models
#DATASET_DIR=/media/dataHD3/kpugdeet/inceptionModel/data/flowers
#CHECKPOINT_FILE=/media/dataHD3/kpugdeet/inceptionModel/flowers-models/inception_v4
#CUDA_VISIBLE_DEVICES=0 python ./slim/eval_image_classifier.py \
#    --alsologtostderr \
#    --checkpoint_path=${CHECKPOINT_FILE} \
#    --dataset_dir=${DATASET_DIR} \
#    --dataset_name=flowers \
#    --dataset_split_name=validation \
#    --model_name=inception_v4

CUDA_VISIBLE_DEVICES=0 python ./slim/export_inference_graph.py \
    --alsologtostderr \
    --model_name=inception_v4 \
    --output_file=/media/dataHD3/kpugdeet/inceptionModel/inception_v4_inf_graph.pb