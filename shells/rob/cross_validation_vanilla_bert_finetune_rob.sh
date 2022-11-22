#!/bin/bash

ENCODER_MODEL_DIR=/home1/cxy/MS/models/BERT_Base_trained_on_MSMARCO
CORPUS_DIR=/home1/cxy/COBERT/data/robust
OUTPUT_DIR=/home1/cxy/COBERT/output/robust/5_cross_validation
TOPFILE=/home1/cxy/MS/data/ms_data/train/topfile_50000.csv
OUTPUT_NAME=train_bert_8
DATANAME=top1_tokens.csv
QID_SPLIT_DIR=/home1/cxy/COBERT/data/robust/5_fold_split
#model_type:vanilla or cobert
CUDA_VISIBLE_DEVICES=1 python ../../code/cross_validation_train.py --data_dir $CORPUS_DIR \
                                               --data_name $DATANAME \
                                               --encoder_model $ENCODER_MODEL_DIR \
                                               --task_name robust \
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
                                               --data_seed 3 \
                                               --eval_step 50 \
                                               --save_step 50000 \

wait
echo "finish train bert!"

#MODEL_DIR=/home1/chenxiaoyang/MS/data/origin/output/train/train_qu_ckpt_1001
#DATA_DIR=/home1/chenxiaoyang/MS/data/origin/data/dev/tokens/dev_qu_tokenize_2020_10_01.csv
#OUTPUT_DIR=/home1/chenxiaoyang/MS/data/origin/output/dev/result_score_qu
#REF_PATH=/home1/chenxiaoyang/MS/data/origin/data/dev/qrel0924.txt
#MRR_path=/home1/chenxiaoyang/MS/data/origin/output/dev/result_eval_qu
#
#python ../code/run_inference_scores.py --device 2,4 \
#                   --model_dir $MODEL_DIR \
#                   --data_dir $DATA_DIR \
#		               --task_name msmarco \
#		               --do_dev\
#                   --output_dir $OUTPUT_DIR \
#                   --max_seq_length 256 \
#		               --eval_batch_size 320\
#		               --ref_file $REF_PATH\
#		               --MRR_path $MRR_path
#
#wait
#echo "finish dev qu!"
