#!/bin/bash


MODEL_DIR=/data/cxy/COBERT/output/gov/5_cross_validation
MODELDIR_NAME=train_bert_5_bertadam
DATA_DIR=/data/cxy/COBERT/data/gov
OUTPUT_DIR=/data/cxy/COBERT/output/gov/5_cross_validation
OUTPUT_NAME=dev_bert_5_bertadam
REF_PATH_NDCG=/data/cxy/MS/data/gov_data/qrels
REF_PATH=/data/cxy/MS/data/gov_data/qrels
QID_SPLIT_DIR=/data/cxy/MS/data/gov_data/5_fold_split
DATA_NAME=top1_tokens.csv

python ../../code/cross_inference.py --device 0 \
                   --model_dir $MODEL_DIR \
                   --modeldir_name $MODELDIR_NAME \
                   --outdir_name $OUTPUT_NAME \
                   --data_dir $DATA_DIR \
                   --data_name $DATA_NAME \
                   --model_name vanilla \
		               --task_name gov \
		               --multi_ckpts \
                   --fold 5 \
                   --dev_ratio 1 \
                   --test_ratio 1 \
                   --qid_split_dir $QID_SPLIT_DIR \
		               --do_dev\
                   --output_dir $OUTPUT_DIR \
                   --max_seq_length 256 \
		               --eval_batch_size 64\
		               --data_seed 3 \
		               --ref_file $REF_PATH \
		               --ref_file_ndcg $REF_PATH_NDCG
wait

MODELDIR_NAME=train_bert_5_bertadam/best4test
FINAL_PATH=/data/cxy/COBERT/output/gov/test/5_cross_results
OUTPUT_NAME=test_bert_5_bertadam

python ../../code/cross_inference.py --device 0 \
                   --model_dir $MODEL_DIR \
                   --modeldir_name $MODELDIR_NAME \
                   --outdir_name $OUTPUT_NAME \
                   --data_dir $DATA_DIR \
                   --data_name $DATA_NAME \
                   --model_name vanilla \
		               --task_name gov \
		               --multi_ckpts \
                   --fold 5 \
                   --dev_ratio 1 \
                   --test_ratio 1 \
                   --qid_split_dir $QID_SPLIT_DIR \
		               --do_test\
                   --output_dir $OUTPUT_DIR \
                   --max_seq_length 256 \
		               --eval_batch_size 64\
		               --data_seed 3 \
		               --ref_file $REF_PATH \
		               --final_path $FINAL_PATH