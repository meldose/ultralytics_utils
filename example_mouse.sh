#! /bin/bash

dataset_root=/home/aime/data/object_perception/training_data

datasets="${dataset_root}/general_model_v3/general_model_v3.yaml \
    ${dataset_root}/general_model_v4/general_model_v4.yaml \
    ${dataset_root}/general_model_v5/general_model_v5.yaml \
    ${dataset_root}/general_model_v6/general_model_v6.yaml \
    ${dataset_root}/general_model_v7/general_model_v7.yaml \
    ${dataset_root}/general_model_v12/general_model_v12.yaml"

save_dir=runs/segment/mouse
model_dir=trained_models/mouse
masks_mode=masks
class_names=mouse

python scripts/train_custom.py \
    --data ${datasets} \
    --task segment \
    --model yolo11s-seg \
    --save-dir ${save_dir}/train \
    --model-dir ${model_dir} \
    --vis-period 1 \
    --epochs 10 \
    --masks-mode ${masks_mode} \
    --hyp hyp/extended.yaml \
    --class-names ${class_names}

# Evaluate model
python scripts/evaluate_custom.py \
    --data ${datasets} \
    --save-dir ${save_dir}/val \
    --weights ${save_dir}/train/weights/best.pt \
    --masks-mode ${masks_mode} \
    --class-names ${class_names}

# Evaluate model
python scripts/evaluate_custom.py \
    --data ${datasets} \
    --save-dir ${save_dir}/test \
    --weights ${save_dir}/train/weights/best.pt \
    --masks-mode ${masks_mode} \
    --class-names ${class_names}

# Infer model
python scripts/infer_custom.py \
    --data ${dataset_root}/general_model_v3/real/test/rgb \
    --save-dir ${save_dir}/infer \
    --weights ${save_dir}/train/weights/best.pt \
    --masks-mode ${masks_mode} \
    --color-mode class \
    --class-names ${class_names}
