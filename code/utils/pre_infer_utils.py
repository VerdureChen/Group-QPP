import os
from tqdm import tqdm
import linecache
import datetime
import random
import argparse
import logging
import sys

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

def select_list_top1(output_path, pre_result_path):
    '''
    与前面的先选出训练数据再选top1不同，这里由于保留了所有的preinfer分数，
    需要先根据preinfer的结果选出top1的信息，（本函数）
    再根据top1的信息去tokenize数据中选择训练中需要用到的部分，并对相关信息条目进行筛选（下一步工作）
    :param fold_num:
    :return:
    '''
    input_result_files = [
        pre_result_path
        ]
    output_top1_result_file = os.path.join(output_path, 'top1result.txt')
    with open(output_top1_result_file, 'w', encoding='utf-8') as outf:
        count = 0
        results = {}
        for input_file in input_result_files:
            with open(input_file, 'r', encoding='utf-8') as in_results:
                num_examples = int(len(linecache.getlines(input_file)))
                for i, line in enumerate(tqdm(in_results, total=num_examples, desc="Tokenize examples")):
                    qid, _, docid, rank, score, _ = line.strip().split(' ')
                    psg_id, psg_bias = docid.rsplit('_')
                    topid, posid = qid, psg_id
                    topdict = results.setdefault(topid, {})
                    if posid not in topdict:
                        outf.write(line)
                        topdict[posid] = float(score)
                        count = count + 1
                    else:
                        continue
        print(f'finish deal with {output_top1_result_file}, nowfile has {str(count)} examples.')
        return output_top1_result_file


def generate_top1_tokens(top1_result_file, pre_token_path, initial_run_file):
    result_file =top1_result_file
    preinfer_tokens_files=[pre_token_path]
    output_path = os.path.join(os.path.dirname(pre_token_path), 'top1_pre_tokens.csv')

    linecount = 0
    with open(output_path, 'w', encoding='utf-8') as outf:
        with open(result_file, 'r', encoding='utf-8') as results,\
             open(initial_run_file, 'r', encoding='utf-8') as initial_run:
            num_examples = int(len(linecache.getlines(initial_run_file)))
            run_dict={}
            for i, line in enumerate(tqdm(initial_run, total=num_examples, desc="Initial run")):
                top_id, _, doc_id, rank, _, _ = line.strip().split(' ')
                run_dict.setdefault(top_id, {})
                run_dict[top_id][doc_id]=rank
            pre_tokens = {}
            for preinfer_file in preinfer_tokens_files:
                with open(preinfer_file, 'r', encoding='utf-8') as pre:
                    num_examples = int(len(linecache.getlines(preinfer_file)))
                    print('number of token-examples: ', str(num_examples))
                    for i, line in enumerate(tqdm(pre, total=num_examples, desc="Pre-infer examples")):

                        tokens = line.strip().split(',')
                        qid=tokens[0]
                        did=tokens[1]
                        bias=tokens[2]
                        pos_input_id=tokens[3]
                        attn=tokens[4]
                        seg=tokens[5]
                        qrel_score=tokens[6]
                        position = run_dict[qid][did]
                        pre_tokens.setdefault(qid,{})
                        pre_tokens[qid][did+'_'+bias]=qid + "," + did + "," + position + "," + \
                                                      pos_input_id + "," + attn + "," + seg + "," + str(qrel_score) + "\n"


            num_examples = int(len(linecache.getlines(result_file)))
            print('number of examples: ', str(num_examples))
            for i, line in enumerate(tqdm(results, total=num_examples, desc="Tokenize examples")):
                qid, _, docid, rank, score, _ = line.strip().split(' ')
                psg_id, psg_bias = docid.rsplit('_')
                # if qid not in q_dict:
                #     if qid not in invalid_list:
                #         invalid_list.append(qid)
                #     continue
                outf.write(pre_tokens[qid][docid])
                linecount+=1
    return linecount, output_path