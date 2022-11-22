"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function
from torch import nn
import argparse
import linecache
import csv
import logging
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
import torch.nn.functional as F
import math

from torch.nn import CrossEntropyLoss, MSELoss, MaxPool1d, AvgPool1d

import transformers
from transformers import BertForTokenClassification, BertForSequenceClassification, BertModel, BertConfig, \
    BertTokenizer, AdamW, get_linear_schedule_with_warmup
from itertools import cycle
# from transformer.modeling2 import BertForSequenceClassification, BertEmbeddings
# # from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
# from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.dataloader import get_labels, output_modes, get_rank_task_dataloader
from model.cobert import COBERT
from model.ql_cobert import QL_COBERT
from model.bertbase import BERT

# os.environ['CUDA_VISIBLE_DEVICES'] = '1, 3'
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


def get_order_dict(topfile):
    qpp_id_order = {}
    qpp_dict = {}
    with open(topfile, 'r', encoding='utf-8') as qpp_file:
        for line in qpp_file:
            qid, score = line.strip().split(' ')
            qpp_dict[qid] = float(score)
    order = sorted(qpp_dict.items(), key=lambda item: item[1], reverse=True)
    for i,item in enumerate(order):
        qpp_id_order[item[0]] = i
    return qpp_id_order

def get_score_dict(ql_file):
    ql_score_dict = {}
    with open(ql_file, 'r', encoding='utf-8') as qlf:
        for line in qlf:
            qid, _, did, _, score, _ = line.split(' ')
            ql_score_dict.setdefault(qid, {})
            ql_score_dict[qid][did] = float(score)
    return ql_score_dict


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        # logger.info("***** Eval results *****")
        # writer.write("***** Train loss *****\n")
        # for key in sorted(result.keys()):
        for key in result.keys():
            # logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("-----------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--data_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The csv data name.")
    parser.add_argument("--qpp_methods",
                        nargs='*',
                        default=[],
                        help="qpp_file list.")
    parser.add_argument("--random",
                        action='store_true',
                        help="Whether to use random sampler.")
    parser.add_argument("--pos",
                        action='store_true',
                        help="Whether to use pos sampler.")
    parser.add_argument("--samplers",
                        default=None,
                        type=int,
                        help="The number of samplers in cross-validation.")
    parser.add_argument("--encoder_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The encoder model dir.")
    parser.add_argument("--groupwise_model",
                        default=None,
                        type=str,
                        help="The groupwise model dir.")
    parser.add_argument("--attn_model",
                        default=None,
                        type=str,
                        help="The attn model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--ref_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The path of ref_file.")
    parser.add_argument("--metric",
                        default=None,
                        type=str,
                        required=True,
                        help="The metric to fit.")
    parser.add_argument("--ql_ranking_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The path of ranking_file.")
    # parser.add_argument("--qpp_method",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="Clarity, ISD, NQC, QF, SD, SMV, UEF, WIG")
    parser.add_argument("--trec_eval_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The path of initial trec_eval.")
    parser.add_argument("--outdir_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The output dir name.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The model type to train.")
    parser.add_argument("--fold",
                        default=None,
                        type=int,
                        help="The number of folds in cross-validation.")
    parser.add_argument("--dev_ratio",
                        default=None,
                        type=int,
                        help="The number of dev folds in cross-validation.")
    parser.add_argument("--test_ratio",
                        default=None,
                        type=int,
                        help="The number of test folds in cross-validation.")
    parser.add_argument("--qid_split_dir",
                        default=None,
                        type=str,
                        help="The qid split file dir.")
    parser.add_argument("--label_type",
                        default=None,
                        type=str,
                        help="The label_type: all,ap or ndcg.")
    parser.add_argument("--qid_file",
                        default=None,
                        type=str,
                        help="The qid file path if you need to split it randomly.")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--top_num",
                        default=0,
                        type=int,
                        help="Total top doc number.")
    parser.add_argument("--rank_num",
                        default=0,
                        type=int,
                        help="Total rank doc number.")
    parser.add_argument("--overlap",
                        default=0,
                        type=int,
                        help="Total overlap number.")
    parser.add_argument("--eval_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=2.0,
                        type=float,
                        help="number of training epochs to perform for one split data.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--data_seed',
                        type=int,
                        default=2,
                        help="seed for data")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--label_num',
                        type=int,
                        default=2,
                        help="label num for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--eval_step',
                        type=int,
                        default=1000)
    parser.add_argument('--save_step',
                        type=int,
                        default=50000)
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    # Prepare  Data
    task_name = args.task_name.lower()
    output_mode = output_modes[task_name]
    label_list = get_labels(task_name.lower())
    num_labels = args.label_num


    # folds = [int(args.fold)]
    folds = list(range(args.fold))
    for fold in folds:
        if args.do_train:

            if args.model_name == 'cobert':
                student_model = COBERT(args.encoder_model, args.groupwise_model, args.attn_model, num_labels, fold=fold)
                logger.info('model_name: cobert')
            elif args.model_name == 'ql_cobert':
                student_model = QL_COBERT(args.encoder_model, args.groupwise_model, args.attn_model, num_labels, fold=fold)
                logger.info('model_name: ql_cobert')
            elif args.model_name == 'poscobert':
                student_model = COBERT(args.encoder_model, args.groupwise_model, args.attn_model, num_labels, fold=fold)
                logger.info('model_name: pos_cobert')
            elif args.model_name == 'vanilla':
                student_model = BERT(args.encoder_model, num_labels=num_labels)
                logger.info('model_name: vanilla-bert')
            else:
                raise NotImplementedError

            if args.gradient_accumulation_steps < 1:
                raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                    args.gradient_accumulation_steps))

            args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
            qpp_file_paths = []
            samplers = []
            order_dicts = []
            last_qpp = ''
            last_dict = {}
            for sn in args.qpp_methods:
                qpp_file_path = os.path.join(args.output_dir, str(fold), 'train', 'bsln', sn, 'best')
                if args.model_name == 'cobert':
                    qpp_file_paths.append(qpp_file_path)
                    order_dict = get_order_dict(qpp_file_path)
                    order_dicts.append(order_dict)
                    last_dict = order_dict
                    samplers.append('cross_cobert')
                last_qpp = qpp_file_path
            if args.random:
                qpp_file_paths.append(last_qpp)
                order_dicts.append(last_dict)
                samplers.append('cross_random')
            if args.pos:
                qpp_file_paths.append(last_qpp)
                order_dicts.append(last_dict)
                samplers.append('pos_cross_cobert')

            num_examples = []
            train_dataloaders = []
            batch_nums = []
            num_examples, train_dataloaders, batch_nums = get_rank_task_dataloader(qpp_file_paths, fold, task_name, 'train', args,
                                                                                 samplers,
                                                                                 batch_size=args.train_batch_size)


            logger.info(f'batch_num:{batch_nums}')
            batch_num_max = max(batch_nums)
            batch_num = batch_num_max*len(batch_nums)
            num_train_optimization_steps = int(
                batch_num / args.gradient_accumulation_steps) * args.num_train_epochs
            student_model.to(device)
            logger.info(f'num examples:{sum(num_examples)}')

            score_dict = get_score_dict(args.ql_ranking_file)
            size = 0
            for n, p in student_model.named_parameters():
                logger.info('n: {}'.format(n))
                size += p.nelement()

            logger.info('Total parameters of student_model: {}'.format(size))

            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", sum(num_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Batch num = %d", batch_num)
            logger.info("  Num steps = %d", num_train_optimization_steps)

            # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            # optimizer_grouped_parameters = [
            #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            #      'weight_decay': 0.01},
            #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            # ]
            # schedule = 'warmup_linear'
            # optimizer = BertAdam(optimizer_grouped_parameters,
            #                      schedule=schedule,
            #                      lr=args.learning_rate,
            #                      warmup=args.warmup_proportion,
            #                      t_total=num_train_optimization_steps)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_train_optimization_steps * args.warmup_proportion,
                                                        num_train_optimization_steps)
            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                student_model, optimizer = amp.initialize(student_model, optimizer, opt_level=args.fp16_opt_level)
                # teacher_model.half()
                logger.info('FP16 is activated, use amp')

            else:
                logger.info('FP16 is not activated, only use BertAdam')

            if n_gpu > 1:
                student_model = torch.nn.DataParallel(student_model)

            # Train and evaluate
            global_step = 0
            # best_dev_acc = 0.0
            tr_loss = 0.

            # Prepare task settings
            if os.path.exists(os.path.join(args.output_dir, str(fold), 'train', args.outdir_name)) \
                    and os.listdir(os.path.join(args.output_dir, str(fold), 'train', args.outdir_name)):
                raise ValueError("Output directory ({}) already exists and is not empty.".format(
                    os.path.join(args.output_dir, str(fold), 'train', args.outdir_name)))

            if not os.path.exists(os.path.join(args.output_dir, str(fold), 'train', args.outdir_name)):
                os.makedirs(os.path.join(args.output_dir, str(fold), 'train', args.outdir_name))
            output_loss_file = os.path.join(args.output_dir, str(fold), 'train', args.outdir_name, "train_loss.txt")

            for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
                student_model.train()
                # nb_tr_examples, nb_tr_steps = 0, 0
                if len(train_dataloaders)==1:
                    train_dataloader = train_dataloaders[0]
                    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                        batch = tuple(t for t in batch)
                        if len(batch) == 8:
                            input_ids, input_mask, segment_ids, query_ids, doc_ids, biass, features, label_ids = batch
                            features = features.to(device)
                        else:
                            input_ids, input_mask, segment_ids, query_ids, doc_ids, biass, label_ids = batch
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        query_ids = query_ids.cpu().numpy()
                        biass = biass.to(device)
                        # doc_ids = doc_ids.cpu()

                        indexs = []
                        ql_scores = []
                        if args.model_name == 'cobert':
                            order_dict = order_dicts[0]
                            for idx in range(len(query_ids)):
                                qid = str(query_ids[idx])
                                if qid in order_dict:
                                    index = order_dict[qid]
                                else:
                                    if args.random:
                                        index = random.randint(0,len(query_ids)-1)
                                    else:
                                        index = random.randint(0,len(order_dict)-1)

                                indexs.append(index)
                                # did = doc_ids[idx]
                            indexs = torch.tensor(indexs, dtype=torch.long).to(device)
                        # if step<10:

                        if args.model_name == 'poscobert':
                            indexs = biass.long()


                        try:
                            student_logits = student_model(input_ids, segment_ids, input_mask, label_ids,
                                                                 indexs, args, ql_scores)
                            # 适配bert-base
                        except RuntimeError as exception:
                            if "out of memory" in str(exception):
                                print("WARNING: out of memory")
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                            else:
                                raise exception

                        if output_mode == "classification":
                            loss = torch.nn.functional.cross_entropy(student_logits, label_ids[int(args.top_num):],
                                                                     reduction='mean')
                        elif output_mode == "regression":
                            # logger.info(label_ids)
                            # logger.info(student_logits.view(-1))
                            loss_mse = MSELoss()
                            loss = loss_mse(student_logits.view(-1), label_ids.view(-1))
                            # cont = 0
                            # for i in range(len(student_logits.view(-1)) - 1):
                            #     for j in range(i+1, len(student_logits.view(-1))):
                            #         loss = loss + max(torch.tensor(0.0, dtype=torch.float, requires_grad=True).to(device),
                            #                           abs(label_ids.view(-1)[i] - label_ids.view(-1)[j]) - torch.sign(
                            #                               label_ids.view(-1)[i] - label_ids.view(-1)[j]) * (
                            #                                       student_logits.view(-1)[i] - student_logits.view(-1)[j]))
                            #         cont+=1
                            # loss = loss/cont

                        if n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        tr_loss += loss.item()

                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            global_step += 1

                        if global_step % args.eval_step == 0:
                            loss = tr_loss / (global_step + 1)

                            result = {}
                            result['global_step'] = global_step
                            result['loss'] = loss

                            result_to_file(result, output_loss_file)

                        if global_step % len(train_dataloader) == 0 or global_step == num_train_optimization_steps:
                            # 适配model
                            logger.info("***** Save model *****")
                            if args.model_name == 'vanilla':
                                checkpoint_name = 'checkpoint-' + str(global_step)
                                # if not args.pred_distill:
                                #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                                output_model_dir = os.path.join(args.output_dir, str(fold), 'train', args.outdir_name,
                                                                checkpoint_name)
                                if not os.path.exists(output_model_dir):
                                    os.makedirs(output_model_dir)
                                student_model.save_pretrained(output_model_dir)
                            else:
                                model_to_save = student_model.module if hasattr(student_model,
                                                                                'module') else student_model
                                checkpoint_name = 'checkpoint-' + str(global_step)
                                output_model_dir = os.path.join(args.output_dir, str(fold), 'train', args.outdir_name,
                                                                checkpoint_name)
                                if not os.path.exists(output_model_dir):
                                    os.makedirs(output_model_dir)
                                torch.save(model_to_save.state_dict(), os.path.join(output_model_dir, 'weights.pt'))
                else:
                    step = 0
                    logger.info(f'num_loaders:{len(train_dataloaders)}')
                    num_loaders = len(train_dataloaders)
                    if len(train_dataloaders) == 3:
                        zips = zip(train_dataloaders[0], cycle(train_dataloaders[1]), cycle(train_dataloaders[2]))
                    elif len(train_dataloaders) == 2:
                        zips = zip(train_dataloaders[0], cycle(train_dataloaders[1]))
                    else:
                        raise NotImplementedError
                    assert args.num_train_epochs!=5
                        # , cycle(train_dataloaders[2]),
                        #        cycle(train_dataloaders[3]), cycle(train_dataloaders[4]))
                    for steps, batchs in tqdm(enumerate(zips), desc="Iteration", ascii=True):
                        for i, batch in enumerate(batchs):
                            batch = tuple(t for t in batch)
                            if len(batch) == 8:
                                input_ids, input_mask, segment_ids, query_ids, doc_ids, biass, features, label_ids = batch
                                features = features.to(device)
                            else:
                                input_ids, input_mask, segment_ids, query_ids, doc_ids, biass, label_ids = batch
                            input_ids = input_ids.to(device)
                            input_mask = input_mask.to(device)
                            segment_ids = segment_ids.to(device)
                            label_ids = label_ids.to(device)
                            query_ids = query_ids.cpu().numpy()
                            biass = biass.to(device)
                            # doc_ids = doc_ids.cpu()
                            order_dict = order_dicts[i]
                            indexs = []
                            for idx in range(len(query_ids)):
                                qid = str(query_ids[idx])
                                if qid in order_dict:
                                    index = order_dict[qid]
                                else:
                                    index = random.randint(0, len(order_dict) - 1)
                                indexs.append(index)
                                # did = doc_ids[idx]
                            if i == 1:
                                random.shuffle(indexs)
                            indexs = torch.tensor(indexs, dtype=torch.long).to(device)

                            ql_scores = []
                            for idx in range(len(query_ids)):
                                qid = str(query_ids[idx])
                                did = str(doc_ids[idx])
                                ql_scores.append(score_dict[qid][did])
                            ql_scores = torch.tensor(ql_scores).unsqueeze(1).to(device)
                            if i == 2:
                                indexs = biass.long()


                            if step < 6:
                                logger.info(indexs)
                            try:
                                student_logits = student_model(input_ids, segment_ids, input_mask, label_ids,
                                                               indexs, args, ql_scores)
                                # 适配bert-base
                            except RuntimeError as exception:
                                if "out of memory" in str(exception):
                                    print("WARNING: out of memory")
                                    if hasattr(torch.cuda, 'empty_cache'):
                                        torch.cuda.empty_cache()
                                else:
                                    raise exception

                            if output_mode == "classification":
                                loss = torch.nn.functional.cross_entropy(student_logits, label_ids[int(args.top_num):],
                                                                         reduction='mean')
                            elif output_mode == "regression":
                                # logger.info(label_ids)
                                # logger.info(student_logits.view(-1))
                                loss_mse = MSELoss()
                                loss = loss_mse(student_logits.view(-1), label_ids.view(-1))
                                # cont = 0
                                # for i in range(len(student_logits.view(-1)) - 1):
                                #     for j in range(i+1, len(student_logits.view(-1))):
                                #         loss = loss + max(torch.tensor(0.0, dtype=torch.float, requires_grad=True).to(device),
                                #                           abs(label_ids.view(-1)[i] - label_ids.view(-1)[j]) - torch.sign(
                                #                               label_ids.view(-1)[i] - label_ids.view(-1)[j]) * (
                                #                                       student_logits.view(-1)[i] - student_logits.view(-1)[j]))
                                #         cont+=1
                                # loss = loss/cont

                            if n_gpu > 1:
                                loss = loss.mean()  # mean() to average on multi-gpu.
                            if args.gradient_accumulation_steps > 1:
                                loss = loss / args.gradient_accumulation_steps

                            if args.fp16:
                                with amp.scale_loss(loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                loss.backward()

                            tr_loss += loss.item()

                            if (step + 1) % args.gradient_accumulation_steps == 0:
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()
                                global_step += 1
                                step += 1

                            if global_step % args.eval_step == 0:
                                loss = tr_loss / (global_step + 1)

                                result = {}
                                result['global_step'] = global_step
                                result['loss'] = loss

                                result_to_file(result, output_loss_file)

                            if global_step % len(train_dataloaders[0]) == 0 or global_step == num_train_optimization_steps:
                                # 适配model
                                logger.info(f"***** Save model {epoch}*****")
                                if args.model_name == 'vanilla':
                                    checkpoint_name = 'checkpoint-' + str(global_step)
                                    # if not args.pred_distill:
                                    #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                                    output_model_dir = os.path.join(args.output_dir, str(fold), 'train', args.outdir_name,
                                                                    checkpoint_name)
                                    if not os.path.exists(output_model_dir):
                                        os.makedirs(output_model_dir)
                                    student_model.save_pretrained(output_model_dir)
                                else:
                                    model_to_save = student_model.module if hasattr(student_model,
                                                                                    'module') else student_model
                                    checkpoint_name = 'checkpoint-' + str(global_step)
                                    output_model_dir = os.path.join(args.output_dir, str(fold), 'train', args.outdir_name,
                                                                    checkpoint_name)
                                    if not os.path.exists(output_model_dir):
                                        os.makedirs(output_model_dir)
                                    torch.save(model_to_save.state_dict(), os.path.join(output_model_dir, 'weights.pt'))


if __name__ == "__main__":
    main()
