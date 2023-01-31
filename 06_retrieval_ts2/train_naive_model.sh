#!/bin/bash -i

CLIP_ROOT=<your_clip_root>
ANNOTATION_ROOT=<your_annotation_root>
OUTPUT_ROOT=<your_model_root>

# CLIP_ROOT=../RTS_clips
# ANNOTATION_ROOT=../RTS_annos
# OUTPUT_ROOT=../RTS_models

if [ ! -d "$OUTPUT_ROOT/naive" ]; then
  mkdir -p "$OUTPUT_ROOT/naive"
fi

python -m torch.distributed.launch --nproc_per_node=1 \
main_task_retrieval.py \
--train_csv "${ANNOTATION_ROOT}/train.csv" \
--val_csv   "${ANNOTATION_ROOT}/naive_test.csv" \
--data_path "${ANNOTATION_ROOT}/naive_captions.json" \
--features_path "${CLIP_ROOT}/videos" \
--output_dir    "${OUTPUT_ROOT}/naive" \
\
--do_train \
--eval_in_train \
--num_thread_reader=4 \
--epochs=5 \
--batch_size=32 \
--batch_size_val 8 \
--n_display=10 \
--pretrained_clip_name ViT-B/32 \
\
--cross_num_hidden_layers 4 \
--lr 1e-4 --max_words 32 --max_frames 12  \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf
