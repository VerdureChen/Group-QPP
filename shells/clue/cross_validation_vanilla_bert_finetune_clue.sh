#!/bin/bash

ENCODER_MODEL_DIR=/home1/cxy/MS/models/BERT_Base_trained_on_MSMARCO
CORPUS_DIR=/home1/cxy/COBERT/data/clue
OUTPUT_DIR=/home1/cxy/COBERT/output/clue/5_cross_validation
TOPFILE=/home1/cxy/MS/data/ms_data/train/topfile_50000.csv
OUTPUT_NAME=train_bert_8
DATANAME=top1_tokens.csv
QID_SPLIT_DIR=/home1/cxy/COBERT/data/clue/5_fold_split
#model_type:vanilla or cobert
CUDA_VISIBLE_DEVICES=2 python ../../code/cross_validation_train.py --data_dir $CORPUS_DIR \
                                               --data_name $DATANAME \
                                               --encoder_model $ENCODER_MODEL_DIR \
                                               --task_name clue \
                                               --output_dir $OUTPUT_DIR \
                                               --outdir_name $OUTPUT_NAME \
                                               --topfile $TOPFILE \
                                               --model_name vanilla \
                                               --max_seq_length 256 \
                                               --fold 5 \
                                               --dev_ratio 1 \
                                               --test_ratio 1 \
                                               --qid_split_dir $QID_SPLIT_DIR \
                                               --do_train \
                                               --fp16 \
                                               --do_lower_case \
                                               --train_batch_size 64 \
                                               --learning_rate 3e-06 \
                                               --num_train_epochs 8.0 \
                                               --seed 42 \
                                               --data_seed 5 \
                                               --eval_step 50 \
                                               --save_step 50000 \

wait
echo "finish train bert!"

