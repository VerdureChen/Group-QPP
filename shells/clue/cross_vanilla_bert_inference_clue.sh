#!/bin/bash


MODEL_DIR=/home1/cxy/COBERT/output/clue/5_cross_validation
MODELDIR_NAME=train_bert_8
DATA_DIR=/home1/cxy/COBERT/data/clue
OUTPUT_DIR=/home1/cxy/COBERT/output/clue/5_cross_validation
OUTPUT_NAME=dev_bert_8
REF_PATH_NDCG=/home1/cxy/COBERT/data/clue/qrels
REF_PATH=/home1/cxy/COBERT/data/clue/qrels
QID_SPLIT_DIR=/home1/cxy/COBERT/data/clue/5_fold_split
DATA_NAME=top1_tokens.csv

python ../../code/cross_inference.py --device 1 \
                   --model_dir $MODEL_DIR \
                   --modeldir_name $MODELDIR_NAME \
                   --outdir_name $OUTPUT_NAME \
                   --data_dir $DATA_DIR \
                   --data_name $DATA_NAME \
                   --model_name vanilla \
		               --task_name clue \
		               --multi_ckpts \
		               --ndcg_metric ndcg_cut_20 \
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

MODELDIR_NAME=train_bert_8/best4test
FINAL_PATH=/home1/cxy/COBERT/output/clue/test/5_cross_results
OUTPUT_NAME=test_bert_8

python ../../code/cross_inference.py --device 1 \
                   --model_dir $MODEL_DIR \
                   --modeldir_name $MODELDIR_NAME \
                   --outdir_name $OUTPUT_NAME \
                   --data_dir $DATA_DIR \
                   --data_name $DATA_NAME \
                   --model_name vanilla \
		               --task_name clue \
		               --multi_ckpts \
		               --ndcg_metric ndcg_cut_20 \
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