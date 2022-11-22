#!/bin/bash

#sleep 2h
##roberta-base
#ENCODER_MODEL_DIR=sentence-transformers/msmarco-roberta-base-v3
#GROUPWISE_MODEL_DIR=/home1/cxy/MS/models/2-512/uncased_L-4_H-768_A-12
#ATTN_MODEL_DIR=/home1/cxy/MS/models/2-512/uncased_L-2_H-768_A-12
##bert-small
#ENCODER_MODEL_DIR=/home1/cxy/MS/models/2-512/vanilla_bert_small_on_MSMARCO
#GROUPWISE_MODEL_DIR=/home1/cxy/MS/models/2-512/uncased_L-4_H-512_A-8
#ATTN_MODEL_DIR=/home1/cxy/MS/models/2-512/uncased_L-2_H-512_A-8
##bert-large
ENCODER_MODEL_DIR=/home1/cxy/MS/models/bert-large/BERT_Large_trained_on_MSMARCO
GROUPWISE_MODEL_DIR=/home1/cxy/MS/models/bert-large/wwm_uncased_L-24_H-1024_A-16-4
ATTN_MODEL_DIR=/home1/cxy/MS/models/bert-large/wwm_uncased_L-24_H-1024_A-16-2
#bert-base
#ENCODER_MODEL_DIR=/home1/cxy/MS/models/BERT_Base_trained_on_MSMARCO
#GROUPWISE_MODEL_DIR=/home1/cxy/MS/models/2-512/uncased_L-4_H-768_A-12
#ATTN_MODEL_DIR=/home1/cxy/MS/models/2-512/uncased_L-2_H-768_A-12

CORPUS_DIR=/home1/cxy/QPP/data/robust
OUTPUT_DIR=/home1/cxy/QPP/output2/robust/30_cross_validation
RANK_FILE=/home1/cxy/QPP/data/robust/run.ql.txt
REF_FILE=/home1/cxy/QPP/data/robust/qrels
DATANAME=top1_pre_tokens.csv
QID_SPLIT_DIR=/home1/cxy/QPP/data/robust/30_split
QID_FILE_DIR=/home1/cxy/QPP/data/robust/qids
TREC_EVAL_PATH=/home1/cxy/QPP/code/bin/trec_eval

array=(25)
# shellcheck disable=SC2068
for element in ${array[@]}
do

TRAIN_OUTPUT_NAME=train_cobert_30_0929_ap_100_random_ISD_POS2
LABEL_TYPE=ap
MODEL_NAME=cobert
TRAIN_RANK_NUM=100
BATCH=64
#model_type:vanilla or cobert
CUDA_VISIBLE_DEVICES=1 python ../../code/cross_validation_train.py --data_dir $CORPUS_DIR \
                                               --data_name $DATANAME \
                                               --encoder_model $ENCODER_MODEL_DIR \
                                               --groupwise_model $GROUPWISE_MODEL_DIR \
                                               --attn_model $ATTN_MODEL_DIR \
                                               --task_name robust \
                                               --trec_eval_path $TREC_EVAL_PATH \
                                               --output_dir $OUTPUT_DIR \
                                               --outdir_name $TRAIN_OUTPUT_NAME \
                                               --ref_file $REF_FILE \
                                               --ql_ranking_file $RANK_FILE \
                                               --model_name $MODEL_NAME \
                                               --max_seq_length 256 \
                                               --fold 30 \
                                               --label_num 2 \
                                               --metric map \
                                               --qid_split_dir $QID_SPLIT_DIR \
                                               --qid_file $QID_FILE_DIR \
                                               --do_train \
                                               --fp16 \
                                               --do_lower_case \
                                               --train_batch_size $BATCH \
                                               --seed 42 \
                                               --eval_step 50 \
                                               --save_step 50000 \
                                               --label_type $LABEL_TYPE \
                                               --rank_num $TRAIN_RANK_NUM \
                                               --learning_rate 1e-5 \
                                               --num_train_epochs 2.0 \
                                               --random \
                                               --pos \
                                               --qpp_methods ISD \
                                               --data_seed 96\ >../../logs/train/rob/cobert_noattn_100.out 2>&1 &

wait
echo "finish train cobert!"



MODEL_DIR=/home1/cxy/QPP/output2/robust/30_cross_validation
OUTPUT_DIR=/home1/cxy/QPP/output2/robust/30_cross_validation
FINAL_PATH=/home1/cxy/QPP/output2/robust/score


DEV_OUTPUT_NAME=dev_cobert_30_1021_rqid_base_nolbl_100_top_$element
DEV_RANK_NUM=$element

python ../../code/cross_inference.py --device 3 \
                   --model_dir $MODEL_DIR \
                   --modeldir_name $TRAIN_OUTPUT_NAME \
                   --outdir_name $DEV_OUTPUT_NAME \
                   --data_dir $CORPUS_DIR \
                   --data_name $DATANAME \
                   --encoder_model $ENCODER_MODEL_DIR \
                   --groupwise_model $GROUPWISE_MODEL_DIR \
                   --attn_model $ATTN_MODEL_DIR \
		               --multi_ckpts \
                   --top_num 0 \
                   --overlap 0 \
                   --fold 30 \
                   --ql_ranking_file $RANK_FILE \
                   --trec_eval_path $TREC_EVAL_PATH \
                   --final_path $FINAL_PATH \
                   --qid_split_dir $QID_SPLIT_DIR \
		               --do_dev\
                   --output_dir $OUTPUT_DIR \
                   --max_seq_length 256 \
		               --eval_batch_size $BATCH \
		               --ref_file $REF_FILE \
		               --metric ap \
                   --model_name $MODEL_NAME \
		               --task_name robust \
		               --label_type $LABEL_TYPE \
		               --label_num 2 \
                   --rank_num $DEV_RANK_NUM \
                   --qpp_method ISD \
		               --data_seed 95\ >../../logs/dev/rob/cobert_rqid_base_100_top_$element.out 2>&1 &



TEST_OUTPUT_NAME=test_cobert_30_1021_rqid_base_nolbl_100_top_$element
TEST_RANK_NUM=$element

python ../../code/cross_inference.py --device 3 \
                   --model_dir $MODEL_DIR \
                   --modeldir_name $TRAIN_OUTPUT_NAME \
                   --outdir_name $TEST_OUTPUT_NAME \
                   --data_dir $CORPUS_DIR \
                   --data_name $DATANAME \
                   --encoder_model $ENCODER_MODEL_DIR \
                   --groupwise_model $GROUPWISE_MODEL_DIR \
                   --attn_model $ATTN_MODEL_DIR \
		               --multi_ckpts \
                   --top_num 0 \
                   --overlap 0 \
                   --fold 30 \
                   --ql_ranking_file $RANK_FILE \
                   --trec_eval_path $TREC_EVAL_PATH \
                   --final_path $FINAL_PATH \
                   --qid_split_dir $QID_SPLIT_DIR \
		               --do_test\
                   --output_dir $OUTPUT_DIR \
                   --max_seq_length 256 \
		               --eval_batch_size $BATCH \
		               --ref_file $REF_FILE \
		               --metric ap \
                   --model_name $MODEL_NAME\
		               --task_name robust \
		               --label_type $LABEL_TYPE \
		               --label_num 2 \
                   --rank_num $TEST_RANK_NUM \
                   --qpp_method ISD \
		               --data_seed 94\ >../../logs/test/rob/cobert_rqid_base_100_top_$element.out 2>&1 &

wait
echo "finish dev and test cobert!"


FINAL_PATH=/home1/cxy/QPP/output2/robust/score/inter

python ../../code/inter.py --dev_outdir_name $DEV_OUTPUT_NAME \
                   --test_outdir_name $TEST_OUTPUT_NAME \
                   --fold 30 \
                   --ql_ranking_file $RANK_FILE \
                   --trec_eval_path $TREC_EVAL_PATH \
                   --final_path $FINAL_PATH \
                   --output_dir $OUTPUT_DIR \
                   --qpp_method ISD \
		               --ref_file $REF_FILE

wait
echo "finish interpolate cobert!"

done