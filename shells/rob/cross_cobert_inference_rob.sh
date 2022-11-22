#!/bin/bash
# sleep 5h
#ENCODER_MODEL_DIR=/home1/cxy/QPP/qpp_msmarco
ENCODER_MODEL_DIR=/home1/cxy/MS/models/BERT_Base_trained_on_MSMARCO

GROUPWISE_MODEL_DIR=/home1/cxy/MS/models/2-512/uncased_L-4_H-768_A-12
ATTN_MODEL_DIR=/home1/cxy/MS/models/2-512/uncased_L-2_H-768_A-12
MODEL_DIR=/home1/cxy/QPP/output2/robust/30_cross_validation

MODELDIR_NAME=train_cobert_30_0929_ap_fused_ISD_100_batch
DATA_DIR=/home1/cxy/QPP/data/robust
OUTPUT_DIR=/home1/cxy/QPP/output2/robust/30_cross_validation

OUTPUT_NAME=dev_cobert_30_0929_ap_fused_ISD_100_batch_25
REF_PATH=/home1/cxy/QPP/data/robust/qrels
QID_SPLIT_DIR=/home1/cxy/QPP/data/robust/30_split
DATA_NAME=top1_pre_tokens.csv
QL_RANKING_FILE=/home1/cxy/QPP/data/robust/run.ql.txt
TREC_EVAL_PATH=/home1/cxy/QPP/code/bin/trec_eval
FINAL_PATH=/home1/cxy/QPP/output2/robust/score

#python ../../code/cross_inference.py --device 0 \
#                   --model_dir $MODEL_DIR \
#                   --modeldir_name $MODELDIR_NAME \
#                   --outdir_name $OUTPUT_NAME \
#                   --data_dir $DATA_DIR \
#                   --data_name $DATA_NAME \
#                   --encoder_model $ENCODER_MODEL_DIR \
#                   --groupwise_model $GROUPWISE_MODEL_DIR \
#                   --attn_model $ATTN_MODEL_DIR \
#		               --multi_ckpts \
#                   --top_num 0 \
#                   --overlap 0 \
#                   --fold 30 \
#                   --ql_ranking_file $QL_RANKING_FILE \
#                   --trec_eval_path $TREC_EVAL_PATH \
#                   --final_path $FINAL_PATH \
#                   --qid_split_dir $QID_SPLIT_DIR \
#		               --do_dev \
#                   --output_dir $OUTPUT_DIR \
#                   --max_seq_length 256 \
#		               --eval_batch_size 64 \
#		               --ref_file $REF_PATH \
#		               --metric ap \
#                   --model_name cobert \
#		               --task_name robust \
#		               --label_type ap \
#		               --label_num 2 \
#                   --rank_num 25 \
#                   --qpp_method ISD \
#		               --data_seed 30\ >../../logs/dev/rob/fuisd_100_25.out 2>&1 &

OUTPUT_NAME=test_cobert_30_0929_ap_fused_ISD_100_batch_25
python ../../code/cross_inference.py --device 0 \
                   --model_dir $MODEL_DIR \
                   --modeldir_name $MODELDIR_NAME \
                   --outdir_name $OUTPUT_NAME \
                   --data_dir $DATA_DIR \
                   --data_name $DATA_NAME \
                   --encoder_model $ENCODER_MODEL_DIR \
                   --groupwise_model $GROUPWISE_MODEL_DIR \
                   --attn_model $ATTN_MODEL_DIR \
		               --multi_ckpts \
                   --top_num 0 \
                   --overlap 0 \
                   --fold 30 \
                   --ql_ranking_file $QL_RANKING_FILE \
                   --trec_eval_path $TREC_EVAL_PATH \
                   --final_path $FINAL_PATH \
                   --qid_split_dir $QID_SPLIT_DIR \
		               --do_test \
                   --output_dir $OUTPUT_DIR \
                   --max_seq_length 256 \
		               --eval_batch_size 64 \
		               --ref_file $REF_PATH \
		               --metric ap \
                   --model_name cobert \
		               --task_name robust \
		               --label_type ap \
		               --label_num 2 \
                   --rank_num 25 \
                   --qpp_method ISD \
		               --data_seed 31\ >../../logs/dev/rob/fuisd_100_25.out 2>&1 &

wait