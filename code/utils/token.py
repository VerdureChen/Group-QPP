from transformers import BertTokenizer
import os
import csv
from tqdm import tqdm
import datetime
import linecache
from collections import defaultdict
import logging
import sys

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True, do_lower_case=True)

def ms_bert_tokenize():
    train_data_raw = r'/data/cxy/msmarco_passage/triples.train.small.tsv'
    output_train_data_qb = '/data/cxy/MS/data/ms_data/train/tokens/triple_tokens.csv'
    num_examples = int(len(linecache.getlines(train_data_raw)))
    print('number of examples: ', str(num_examples))

    with open(train_data_raw, 'r', encoding='utf-8') as f, \
            open(output_train_data_qb, 'w', encoding='utf-8') as out_qb:
        count = 0
        for line in tqdm(f, total=num_examples, desc="Tokenize examples"):
            qtext, pos_body, ujg_body = line.rstrip().split('\t')
            q_id, pos_id, ujg_id = ['1','1','1']
            pos_bias = '0'

            pos_input_id, attn, seg = get_tokens(qtext, pos_body)
            out_qb.write(q_id + "," + pos_id + "," + pos_bias + "," +
                         pos_input_id + "," + attn + "," + seg + "," + str(1) + "\n")
            neg_input_id, attn, seg = get_tokens(qtext, ujg_body)
            out_qb.write(q_id + "," + ujg_id + "," + pos_bias + "," +
                         neg_input_id + "," + attn + "," + seg + "," + str(0) + "\n")
            #
            if count < 2:
                logger.info(f'pid:{pos_id}, \nnegid：{ujg_id}, \np_input:{pos_input_id}, \nneg_input:{neg_input_id}')

            count = count + 2
        print('total tokenize number: {}'.format(str(count)))


def gov_clue_bert_tokenize(train_data_raw, output_train_data_qb):

    num_examples = int(len(linecache.getlines(train_data_raw)))
    print('number of examples: ', str(num_examples))

    with open(train_data_raw, 'r', encoding='utf-8') as f, \
            open(output_train_data_qb, 'w', encoding='utf-8') as out_qb:
        count = 0
        for line in tqdm(f, total=num_examples, desc="Tokenize examples"):
            top_id, qtext, pos_docid, pos_body, pos_bias, qrel_score = line.rstrip().split('\t')

            pos_input_id, attn, seg = get_tokens(qtext, pos_body)
            out_qb.write(top_id + "," + pos_docid + "," + pos_bias + "," +
                         pos_input_id + "," + attn + "," + seg + "," + str(qrel_score) + "\n")
            #
            if count < 2:
                logger.info(f'top_id:{top_id}, \npos_id：{pos_docid}, \np_input:{pos_input_id}, \nqrel:{str(qrel_score)}')

            count = count + 1
        print('total tokenize number: {}'.format(str(count)))

def rob_bert_tokenize(train_data_raw, output_train_data_qb):

    num_examples = int(len(linecache.getlines(train_data_raw)))
    print('number of examples: ', str(num_examples))

    with open(train_data_raw, 'r', encoding='utf-8') as f, \
            open(output_train_data_qb, 'w', encoding='utf-8') as out_qb:
        count = 0
        for line in tqdm(f, total=num_examples, desc="Tokenize examples"):
            top_id, qtext, pos_docid, pos_title, pos_body, pos_bias, qrel_score = line.rstrip().split('\t')

            pos_input_id, attn, seg = get_tokens(qtext, pos_title+pos_body)
            out_qb.write(top_id + "," + pos_docid + "," + pos_bias + "," +
                         pos_input_id + "," + attn + "," + seg + "," + str(qrel_score) + "\n")
            #
            if count < 2:
                logger.info(f'top_id:{top_id}, \npos_id：{pos_docid}, \np_input:{pos_input_id}, \nqrel:{str(qrel_score)}')

            count = count + 1
        print('total tokenize number: {}'.format(str(count)))


def get_tokens(sent_1, sent_2):
    sentence_a = sent_1
    sentence_b = sent_2
    encoded = tokenizer.encode_plus(
        text=sentence_a,  # the sentence to be encoded
        text_pair=sentence_b,
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=256,  # maximum length of a sentence
        pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask=True,  # Generate the attention mask
        return_tensors='pt',  # ask the function to return PyTorch tensors
        truncation=True,
    )
    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']
    seg_mask = encoded['token_type_ids']
    input_id = ' '.join([str(item) for item in input_ids.tolist()[0]])
    attn = ' '.join([str(item) for item in attn_mask.tolist()[0]])
    seg = ' '.join([str(item) for item in seg_mask.tolist()[0]])
    return input_id, attn, seg



