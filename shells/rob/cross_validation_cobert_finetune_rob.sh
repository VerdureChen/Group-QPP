#!/bin/bash
#INITIAL_QPP_PATH=/home1/cxy/QPP/data/robust/clarity.txt
# ENCODER_MODEL_DIR=/home1/cxy/QPP/qpp_msmarco
ENCODER_MODEL_DIR=/home1/cxy/MS/models/BERT_Base_trained_on_MSMARCO
GROUPWISE_MODEL_DIR=/home1/cxy/MS/models/2-512/uncased_L-4_H-768_A-12
ATTN_MODEL_DIR=/home1/cxy/MS/models/2-512/uncased_L-2_H-768_A-12
CORPUS_DIR=/home1/cxy/QPP/data/robust
OUTPUT_DIR=/home1/cxy/QPP/output2/robust/30_cross_validation
RANK_FILE=/home1/cxy/QPP/data/robust/run.ql.txt
REF_FILE=/home1/cxy/QPP/data/robust/qrels
OUTPUT_NAME=train_cobert_30_0929_ap_fused_ISD_100_batch_random_6epo
DATANAME=top1_pre_tokens.csv
QID_SPLIT_DIR=/home1/cxy/QPP/data/robust/30_split
QID_FILE_DIR=/home1/cxy/QPP/data/robust/qids
TREC_EVAL_PATH=/home1/cxy/QPP/code/bin/trec_eval
#model_type:vanilla or cobert
CUDA_VISIBLE_DEVICES=0 python ../../code/cross_validation_train.py --data_dir $CORPUS_DIR \
                                               --data_name $DATANAME \
                                               --pos \
                                               --random \
                                               --qpp_methods ISD \
                                               --encoder_model $ENCODER_MODEL_DIR \
                                               --groupwise_model $GROUPWISE_MODEL_DIR \
                                               --attn_model $ATTN_MODEL_DIR \
                                               --task_name robust \
                                               --trec_eval_path $TREC_EVAL_PATH \
                                               --output_dir $OUTPUT_DIR \
                                               --outdir_name $OUTPUT_NAME \
                                               --ref_file $REF_FILE \
                                               --ql_ranking_file $RANK_FILE \
                                               --model_name cobert \
                                               --max_seq_length 256 \
                                               --fold 30 \
                                               --label_num 2 \
                                               --label_type ap \
                                               --rank_num 100 \
                                               --metric map \
                                               --qid_split_dir $QID_SPLIT_DIR \
                                               --qid_file $QID_FILE_DIR \
                                               --do_train \
                                               --fp16 \
                                               --do_lower_case \
                                               --train_batch_size 64 \
                                               --learning_rate 1e-5 \
                                               --num_train_epochs 2.0 \
                                               --seed 42 \
                                               --data_seed 14 \
                                               --eval_step 50 \
                                               --save_step 50000 \

wait
echo "finish train cobert!"


