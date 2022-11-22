import sys
import os
import logging
import glob
import torch
import numpy as np
import linecache
import random
import pandas as pd
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tempfile import TemporaryDirectory
from pathlib import Path
import pickle
import math
import gc
from tqdm import tqdm
import sys
from .cosampler import COBERTSampler, COBERTBatchSampler, cross_COBERTSampler, cross_RandomSampler,\
    posCOBERTBatchSampler, cross_posCOBERTSampler
from utils.cross_validation_utils import split_qids_to_fold, get_fold_qids
from evaluation.evaluation import evaluate_trec_per_query

logger = logging.getLogger(__name__)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "/" + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)

class PregeneratedDataset(Dataset):
    def __init__(self, metric_dict, training_path, set_type, max_seq_length, num_examples, label_type,
                 output_mode='regression', reduce_memory=True, features_file=None, data_seed=2):
        logger.info('training_path: {}'.format(training_path))
        self.seq_len = max_seq_length
        self.output_mode = output_mode
        self.set_type = set_type
        self.metric_dict = metric_dict
        self.num_samples = num_examples
        self.temp_dir = None
        self.working_dir = None
        self.features_file = features_file

        if reduce_memory:
            p = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'cache')
            if data_seed==1:
                #没有其他程序正在运行
                del_file(p)
            input_ids = np.memmap(filename=os.path.join(p, 'input_ids_{}.memmap'.format(str(data_seed))),
                                  mode='w+', dtype=np.int32, shape=(self.num_samples, self.seq_len))
            input_masks = np.memmap(filename=os.path.join(p, 'input_masks_{}.memmap'.format(str(data_seed))),
                                    shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
            segment_ids = np.memmap(filename=os.path.join(p, 'segment_ids_{}.memmap'.format(str(data_seed))),
                                    shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
            label_ids = np.memmap(filename=os.path.join(p, 'label_ids_{}.memmap'.format(str(data_seed))),
                                  shape=(self.num_samples, ), mode='w+', dtype=np.float32)
            label_ids[:] = -1

            query_ids = np.memmap(filename=os.path.join(p, 'query_ids_{}.memmap'.format(str(data_seed))),
                              shape=(self.num_samples, ), mode='w+', dtype=np.int32)
            query_ids[:] = -1

            biass = np.memmap(filename=os.path.join(p, 'biass_{}.memmap'.format(str(data_seed))),
                                  shape=(self.num_samples,), mode='w+', dtype=np.int32)
            biass[:] = -1

            doc_ids = [None]*self.num_samples

        else:
            raise NotImplementedError

        logging.info("Loading training examples.")

        with open(training_path, 'r') as f:
            for i, line in enumerate(tqdm(f, total=self.num_samples, desc="Training examples")):
                tokens = line.strip().split(',')
                input_ids[i] = [int(id) for id in tokens[3].split()]
                input_masks[i] = [int(id) for id in tokens[4].split()]
                segment_ids[i] = [int(id) for id in tokens[5].split()]


                query_ids[i] = int(tokens[0])
                doc_ids[i] = tokens[1]
                biass[i] = int(tokens[2])
                guid = "%s-%s" % (self.set_type, tokens[0]+'-'+tokens[1]+'-'+tokens[2])
                if self.set_type != 'test' and self.set_type != 'dev':
                    if self.output_mode == "classification":
                        label_ids[i] = int(tokens[6])
                    elif self.output_mode == "regression":
                        try:
                            if label_type == 'all':
                                label_ids[i] = float(self.metric_dict[tokens[0]])
                            elif label_type == 'ap':
                                label_ids[i] = self.metric_dict[tokens[0]][biass[i]-1]
                            elif label_type == 'ndcg':
                                label_ids[i] = self.metric_dict[tokens[0]][biass[i]-1]
                            else:
                                raise NotImplementedError
                        except:
                            label_ids[i] = 0
                    else:
                        raise NotImplementedError
                else:
                    label_ids[i] = 0

                # if label_ids[i] != 0 and label_ids[i] != 1:
                #     if label_ids[i] > 0:
                #         label_ids[i] = 1
                #     else:
                #         label_ids[i] = 0

                if i < 2:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % guid)
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids[i]]))
                    logger.info("input_masks: %s" % " ".join([str(x) for x in input_masks[i]]))
                    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids[i]]))
                    logger.info("label: %s" % str(label_ids[i]))

                    logger.info("qid: %s" % str(query_ids[i]))
                    logger.info("docid: %s" % str(doc_ids[i]))
                    logger.info("biass: %s" % str(biass[i]))

        logging.info("Loading complete!")
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.label_ids = label_ids

        self.query_ids = query_ids
        self.doc_ids = doc_ids
        self.biass = biass
        if features_file is not None:
            self.features = torch.load(features_file)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        if self.output_mode == "classification":
            label_id = torch.tensor(self.label_ids[item], dtype=torch.long)
        elif self.output_mode == "regression":
            label_id = torch.tensor(self.label_ids[item], dtype=torch.float)
        else:
            raise NotImplementedError
        # if not self.listmode:
        #     return (torch.tensor(self.input_ids[item], dtype=torch.long),
        #             torch.tensor(self.input_masks[item], dtype=torch.long),
        #             torch.tensor(self.segment_ids[item], dtype=torch.long),
        #             label_id)
        # else:
        if self.features_file is not None:
            return (torch.tensor(self.input_ids[item], dtype=torch.long),
                    torch.tensor(self.input_masks[item], dtype=torch.long),
                    torch.tensor(self.segment_ids[item], dtype=torch.long),
                    self.query_ids[item],
                    self.doc_ids[item],
                    self.biass[item],
                    self.features[item],
                    label_id)
        else:
            return (torch.tensor(self.input_ids[item], dtype=torch.long),
                    torch.tensor(self.input_masks[item], dtype=torch.long),
                    torch.tensor(self.segment_ids[item], dtype=torch.long),
                    self.query_ids[item],
                    self.doc_ids[item],
                    self.biass[item],
                    label_id)


output_modes = {
    "msmarco": "regression",
    "robust": "regression",
    "gov": "regression",
    "clue": "regression",
}

def get_labels(task_name):
    """See base class."""
    if task_name.lower() == "msmarco":
        return ["0", "1"]
    elif task_name.lower() == "robust":
        return ["0", "1"]
    elif task_name.lower() == "gov":
        return ["0", "1"]
    elif task_name.lower() == "clue":
        return ["0", "1"]
    else:
        raise NotImplementedError

def get_ndcg(args):
    qrel = args.ref_file
    ql = args.ql_ranking_file
    lab_dict = {}
    dcg_dict = {}
    rel_dict = {}
    idcg_dict = {}
    ndcg_dict = {}
    with open(qrel, 'r', encoding='utf-8') as qr:
        for line in qr:
            q, _, d, lb = line.strip().split(" ")
            lab_dict.setdefault(q,{})
            lab_dict[q][d] = lb
    with open(ql, 'r', encoding='utf-8') as ql:
        for line in ql:
            qid, _, did, rank, _, _ = line.strip().split(" ")
            dcg_dict.setdefault(qid, [])
            rel_dict.setdefault(qid, [])
            rel = int(lab_dict[qid][did])
            rel_dict[qid].append(rel)
            increse = (math.pow(2, rel)-1)/math.log(int(rank)+1,2)
            if dcg_dict[qid] == []:
                dcg_dict[qid].append(float(increse))
            else:
                pre_sum = sum(dcg_dict[qid])
                dcg_dict[qid].append(pre_sum+increse)
    with open(ql, 'r', encoding='utf-8') as ql:
        for k in rel_dict:
            rel_dict[k] = sorted(rel_dict[k], reverse=True)
        for line in ql:
            qid, _, did, rank, _, _ = line.strip().split(" ")
            idcg_dict.setdefault(qid, [])
            rel = rel_dict[qid][int(rank)-1]
            increse = (math.pow(2, rel) - 1) / math.log(int(rank)+1, 2)
            if idcg_dict[qid] == []:
                idcg_dict[qid].append(float(increse))
            else:
                pre_sum = sum(idcg_dict[qid])
                idcg_dict[qid].append(pre_sum+increse)
    for k in dcg_dict:
        ndcg_dict[k] = [a/b for a,b in zip(dcg_dict[k], idcg_dict[k])]

    return ndcg_dict


def get_ap(args):
    qrel = args.ref_file
    ql = args.ql_ranking_file
    lab_dict = {}
    precision_dict = {}
    rel_dict = {}
    with open(qrel, 'r', encoding='utf-8') as qr:
        for line in qr:
            q, _, d, lb = line.strip().split(" ")
            lab_dict.setdefault(q,{})
            lb = int(lb)
            if lb > 1:
                lb = 1
            lab_dict[q][d] = lb
    with open(ql, 'r', encoding='utf-8') as ql:
        for line in ql:
            qid, _, did, rank, _, _ = line.strip().split(" ")
            precision_dict.setdefault(qid, [])
            rel_dict.setdefault(qid, [])
            try:
                rel = int(lab_dict[qid][did])
            except:
                rel = 0
            rel_dict[qid].append(rel)
            increse = rel*sum(rel_dict[qid][:int(rank)])/int(rank)
            precision_dict[qid].append(float(increse))

    return precision_dict


def get_metric_per_query(args):
        qrel = args.ref_file
        ql = args.ql_ranking_file

        metric_dict, _ = evaluate_trec_per_query(qrel, ql, args.metric, args.trec_eval_path)
        print(metric_dict)
        return metric_dict

def get_rank_task_dataloader(qpp_file_paths, fold_num, task_name, set_name, args, samplers, batch_size=None, dataset2=False):
    if args.fold >1:
        if os.path.exists(args.qid_split_dir) and os.listdir(args.qid_split_dir):
            pass
        else:
            split_qids_to_fold(args.fold, 2, args.qid_file, args.qid_split_dir, args.ref_file)
    if dataset2==False:
        data_name = args.data_name
    else:
        data_name = args.data_name
    output_mode = output_modes[task_name]
    file_dir = os.path.join(args.data_dir, 'tokens/', data_name)
    if args.label_type == 'all':
        metric_dict = get_metric_per_query(args)
    elif args.label_type == 'ndcg':
        metric_dict = get_ndcg(args)
    else:
        metric_dict = get_ap(args)

    num_examples = int(len(linecache.getlines(file_dir)))
    print('number of examples: ', str(num_examples))
    data_seed = int(args.data_seed)
    features_file = None
    if args.model_name=='freeze':
        feature_dirs = os.path.join(args.data_dir, str(fold_num), 'tokens/{}/features'.format(set_name.lower()))
        if os.path.exists(feature_dirs):
            dir_list = [item for item in os.listdir(feature_dirs) if
                        item.startswith('pooled')]
            feature_dir = os.path.join(args.data_dir, str(fold_num), 'tokens/{}/features'.format(set_name.lower()),
                                       dir_list[0])
            features_file = feature_dir
            logger.info(f'feature_path:{feature_dir}')

    dataset = PregeneratedDataset(metric_dict, file_dir, set_name.lower(), args.max_seq_length,
                              num_examples, args.label_type, output_mode, reduce_memory=True, features_file=features_file, data_seed=data_seed)
    num_exampless = []
    dataloaders = []
    batch_nums = []
    for qpp_file_path, sampler in zip(qpp_file_paths, samplers):
        if sampler == 'cross_cobert':
            qids = get_fold_qids(fold_num, args.qid_split_dir, set_name.lower())
            s = cross_COBERTSampler(dataset, batch_size, args, qids, qpp_file_path=qpp_file_path)
            bs = COBERTBatchSampler(s, batch_size)
            dataloader = DataLoader(dataset, batch_sampler=bs)
            num_examples = len(s)
            batch_num = len(bs)
            num_exampless.append(num_examples)
            dataloaders.append(dataloader)
            batch_nums.append(batch_num)
            logger.info('initialize cross_cobert sampler successfully!')
        elif sampler == 'pos_cross_cobert':
            qids = get_fold_qids(fold_num, args.qid_split_dir, set_name.lower())
            s = cross_posCOBERTSampler(dataset, batch_size, args, qids, qpp_file_path=qpp_file_path)
            bs = posCOBERTBatchSampler(s, batch_size)
            dataloader = DataLoader(dataset, batch_sampler=bs)
            num_examples = len(s)
            batch_num = len(bs)
            num_exampless.append(num_examples)
            dataloaders.append(dataloader)
            batch_nums.append(batch_num)
            logger.info('initialize pos_cross_cobert sampler successfully!')
        elif sampler=='cross_random':
            qids = get_fold_qids(fold_num, args.qid_split_dir, set_name.lower())
            s = cross_RandomSampler(dataset, qids, args)
            num_examples = len(s)
            dataloader = DataLoader(dataset, sampler=s, batch_size=batch_size)
            num_exampless.append(num_examples)
            dataloaders.append(dataloader)
            batch_nums.append(len(dataloader))
            logger.info('initialize cross_random sampler successfully!')
    return num_exampless, dataloaders, batch_nums
