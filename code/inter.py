import os
import logging
import argparse
import torch
from tqdm import tqdm
import shutil
import numpy as np
import torch.nn.functional as F
from transformers import BertForSequenceClassification
from utils.dataloader import get_labels, output_modes, get_rank_task_dataloader
from evaluation.ms_marco_eval import compute_metrics_from_files
import datetime
from evaluation.test_trec_eval import validate
from model.cobert import COBERT
from model.bertbase import BERT
from get_corr import get_corr
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir_name1",
                        default=None,
                        type=str,
                        required=True,
                        help="Output dir name.")
    parser.add_argument("--outdir_name2",
                        default=None,
                        type=str,
                        required=True,
                        help="Output dir name.")
    parser.add_argument("--outdir_name3",
                        default=None,
                        type=str,
                        required=True,
                        help="Output dir name.")
    parser.add_argument("--ql_ranking_file",
                        default=None,
                        type=str,
                        required=True,
                        help="QL file.")
    parser.add_argument("--trec_eval_path",
                        default=None,
                        type=str,
                        required=True,
                        help="trec_eval path.")
    parser.add_argument("--final_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Final result path.")
    parser.add_argument("--desc",
                        default=None,
                        type=str,
                        required=True,
                        help="The description.")
    parser.add_argument("--do_dev",
                        action='store_true',
                        help="Whether to run dev.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test.")
    parser.add_argument("--fold",
                        default=None,
                        type=int,
                        help="The number of folds in cross-validation.")
    parser.add_argument("--rank_num",
                        default=0,
                        type=int,
                        help="Rank doc number.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions will be written.")
    parser.add_argument("--ref_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The qrel file path.")


    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # get_final_result(args.fold, args.final_path, args)
    max_list = dev_inter(args)
    test_inter(args, max_list)

def main_inter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_outdir_name",
                        default=None,
                        type=str,
                        required=True,
                        help="Output dir name.")
    parser.add_argument("--test_outdir_name",
                        default=None,
                        type=str,
                        required=True,
                        help="Output dir name.")
    parser.add_argument("--ql_ranking_file",
                        default=None,
                        type=str,
                        required=True,
                        help="QL file.")
    parser.add_argument("--trec_eval_path",
                        default=None,
                        type=str,
                        required=True,
                        help="trec_eval path.")
    parser.add_argument("--final_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Final result path.")
    parser.add_argument("--qpp_method",
                        default=None,
                        type=str,
                        required=True,
                        help="QPP method to inter.")
    parser.add_argument("--fold",
                        default=None,
                        type=int,
                        help="The number of folds in cross-validation.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions will be written.")
    parser.add_argument("--ref_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The qrel file path.")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # get_final_result(args.fold, args.final_path, args)
    max_list = dev_inter(args)
    test_inter(args, max_list)

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    small = np.ones_like(max)*1e-8
    return (data+small - min)/(max+small-min)


def get_score_dict(res_file):
    score_dict = {}
    scores = []
    with open(res_file, 'r', encoding='utf8') as score_fl:
        for line in score_fl:
            qid, score= line.strip().split(' ')
            score_dict.setdefault(qid, 0)
            scores.append(float(score))
        top_score = np.array(scores)
        scal = minmaxscaler(top_score)
        scal = scal.tolist()
    with open(res_file, 'r', encoding='utf8') as score_fl:
        for line, sc in zip(score_fl, scal):
            qid, score = line.split(' ')
            score_dict.setdefault(qid, 0)
            score_dict[qid] = float(sc)
        # for k, v in top_dict.items():
        #     print(f'{k}:{v}')
        logger.info(f'finish construct top_dict, has {len(score_dict)} queries.')
        return score_dict


def dev_inter(args):
    inter_out_path = os.path.join(args.output_dir, '{}', 'dev', args.dev_outdir_name, 'inter_{}'.format(args.qpp_method))
    dev_path = os.path.join(args.output_dir, '{}', 'dev', args.dev_outdir_name,)
    record_file = os.path.join(args.output_dir, '{}', 'dev', args.dev_outdir_name, 'inter_{}'.format(args.qpp_method), 'record.txt')
    qpp_file_path = os.path.join(args.output_dir, '{}', 'train', 'bsln', args.qpp_method, 'best')
    max_list = []
    for i in range(args.fold):
        inter_out_path = inter_out_path.format(i)
        if not os.path.exists(inter_out_path):
            os.makedirs(inter_out_path)
        dev_path = dev_path.format(i)
        record_file = record_file.format(i)
        qpp_file_path = qpp_file_path.format(i)
        lams = [float(item) / 10 for item in range(0, 11, 1)]
        max_score = -1
        with open(record_file, 'w', encoding='utf-8') as record:
            res_files = [item for item in os.listdir(dev_path) if item.startswith('results')]
            qpp_score_dict = get_score_dict(qpp_file_path)
            for res_name in res_files:
                tp = res_name.split('_')[1]
                res_file = os.path.join(dev_path, res_name)
                res_score_dict = get_score_dict(res_file)
                inter_file_name = tp+'_{}'
                for lam in lams:
                    inter_file = os.path.join(inter_out_path, inter_file_name.format(str(lam)))
                    with open(inter_file, 'w', encoding='utf-8') as it_fl:
                        final_dict = {}
                        for topid in qpp_score_dict:
                            final_dict[topid] = lam * res_score_dict[topid] + (1 - lam) * qpp_score_dict[topid]

                        scores = list(sorted(final_dict.items(), key=lambda x: (x[1], x[0]), reverse=True))
                        for i, (qid, score) in enumerate(scores):
                            it_fl.write(f'{qid} {score}\n')

                    path_to_candidate = inter_file
                    path_to_reference = args.ref_file
                    p, k = get_corr(path_to_candidate, path_to_reference, args.ql_ranking_file, args.trec_eval_path,
                                    metric='ap')
                    record.write(f'{inter_file_name.format(str(lam))}, pearson:{p}, kendall:{k}\n')
                    mt = float(p)
                    if mt > max_score:
                        max_score = mt
                        max_outfile = inter_file_name.format(str(lam))
            record.write('MAX FILE:{}, MAX pearson:{}'.format(max_outfile, max_score))
            max_list.append(max_outfile)

        inter_out_path = os.path.join(args.output_dir, '{}', 'dev', args.dev_outdir_name, 'inter_{}'.format(args.qpp_method))
        dev_path = os.path.join(args.output_dir, '{}', 'dev', args.dev_outdir_name, )
        record_file = os.path.join(args.output_dir, '{}', 'dev', args.dev_outdir_name, 'inter_{}'.format(args.qpp_method), 'record.txt')
        qpp_file_path = os.path.join(args.output_dir, '{}', 'train', 'bsln', args.qpp_method, 'best')
    return max_list


def test_inter(args, max_list):
    inter_out_path = os.path.join(args.output_dir, '{}', 'test', args.test_outdir_name, 'inter_{}'.format(args.qpp_method))
    test_path = os.path.join(args.output_dir, '{}', 'test', args.test_outdir_name)
    record_file = os.path.join(args.output_dir, '{}', 'test', args.test_outdir_name, 'inter_{}'.format(args.qpp_method), 'record.txt')
    qpp_file_path = os.path.join(args.output_dir, '{}', 'test', 'bsln', args.qpp_method)
    res_p = []
    res_k = []
    final_record_file = os.path.join(args.final_path,
                                   'record_{}_{}_{}.txt'.format(args.test_outdir_name,
                                                                 datetime.date.today().strftime('%Y_%m_%d'),
                                                                args.qpp_method))
    for i in range(args.fold):
        inter_out_path = inter_out_path.format(i)
        if not os.path.exists(inter_out_path):
            os.makedirs(inter_out_path)
        test_path = test_path.format(i)
        record_file = record_file.format(i)
        qpp_file_path = qpp_file_path.format(i)
        qpp = [item for item in os.listdir(qpp_file_path) if not item.startswith('record')]
        assert len(qpp) == 1
        qpp_file_path = os.path.join(qpp_file_path, qpp[0])

        tp, lam = max_list[i].split('_')
        lam = float(lam)
        dir_list = [item for item in os.listdir(test_path) if item.startswith('results_'+tp)]
        logger.info('*******')
        logger.info(dir_list)
        logger.info('*******')
        def num(ele):
            return int(ele.split('-')[-1].split('_')[0])
        dir_list.sort(key=num, reverse=True)
        res_file = os.path.join(test_path, dir_list[0])

        qpp_score_dict = get_score_dict(qpp_file_path)
        res_score_dict = get_score_dict(res_file)

        with open(record_file, 'w', encoding='utf-8') as record:
            inter_file = os.path.join(inter_out_path, max_list[i])
            with open(inter_file, 'w', encoding='utf-8') as it_fl:
                final_dict = {}
                for topid in qpp_score_dict:
                    final_dict[topid] = lam * res_score_dict[topid] + (1 - lam) * qpp_score_dict[topid]

                scores = list(sorted(final_dict.items(), key=lambda x: (x[1], x[0]), reverse=True))
                for i, (qid, score) in enumerate(scores):
                    it_fl.write(f'{qid} {score}\n')

            path_to_candidate = inter_file
            path_to_reference = args.ref_file
            p, k = get_corr(path_to_candidate, path_to_reference, args.ql_ranking_file, args.trec_eval_path,
                            metric='ap')
            record.write(f'{tp}_{lam}, pearson:{p}, kendall:{k}\n')
            res_p.append(p)
            res_k.append(k)
        inter_out_path = os.path.join(args.output_dir, '{}', 'test', args.test_outdir_name, 'inter_{}'.format(args.qpp_method))
        test_path = os.path.join(args.output_dir, '{}', 'test', args.test_outdir_name)
        record_file = os.path.join(args.output_dir, '{}', 'test', args.test_outdir_name, 'inter_{}'.format(args.qpp_method), 'record.txt')
        qpp_file_path = os.path.join(args.output_dir, '{}', 'test', 'bsln', args.qpp_method)

    with open(final_record_file, 'w', encoding='utf-8') as final:
        p = np.mean(res_p)
        k = np.mean(res_k)
        final.write(f'pearson:{p}, kendall:{k}\n')


def get_final_result(fold, final_path, args):
    record_rank1_file = os.path.join(final_path,
                                     'record_{}_{}.txt'.format(args.desc, datetime.date.today().strftime('%Y_%m_%d')))

    record_list = [record_rank1_file]
    result_str = [args.outdir_name1, args.outdir_name2, args.outdir_name3]
    for j, record_file in enumerate(record_list):
        res_p = {}
        res_k = {}
        with open(record_file, 'w', encoding='utf-8') as record:
            for i in range(fold):
                dir_lists = []
                for method in result_str:
                    result_path = os.path.join(args.output_dir, '{}', 'test/', method)
                    res_path = result_path.format(str(i))
                    dir_list = [item for item in os.listdir(res_path) if item.startswith('results_avg')]


                    def num(ele):
                        return int(ele.split('-')[-1].split('_')[0])

                    dir_list.sort(key=num)
                    logger.info('*******')
                    logger.info(dir_list)
                    logger.info('*******')
                    dir_lists.append(dir_list)
                for epoch in range(2):
                    score_dict = {}
                    inter_path = os.path.join(args.output_dir, str(i), 'test/inter')
                    if not os.path.exists(inter_path):
                        os.makedirs(inter_path)
                    inter_path = os.path.join(inter_path,
                                              'result_wcq_epoch{}_{}.txt'.format(str(epoch),datetime.date.today().strftime('%Y_%m_%d')))
                    for st in range(len(result_str)):
                        fl = dir_lists[st][epoch]
                        result_path = os.path.join(args.output_dir, '{}', 'test/', result_str[st])
                        res_path = result_path.format(str(i))
                        res_file = os.path.join(res_path, fl)
                        scores = []
                        with open(res_file, 'r', encoding='utf-8') as resf:
                            for line in resf:
                                qid, score = line.split(' ')
                                score_dict.setdefault(qid, 0)
                                scores.append(float(score))
                            top_score = np.array(scores)
                            scal = minmaxscaler(top_score)
                            scal = scal.tolist()
                        with open(res_file, 'r', encoding='utf-8') as resf:
                            for line, sc in zip(resf, scal):
                                qid, score = line.split(' ')
                                score_dict.setdefault(qid, 0)
                                score_dict[qid] += float(sc)
                    scores_inter = list(sorted(score_dict.items(), key=lambda x: (x[1], x[0]), reverse=True))
                    with open(inter_path, 'w', encoding='utf-8') as f:
                        for qid, score in scores_inter:
                            f.write(f'{qid} {score}\n')


                    path_to_candidate = inter_path
                    path_to_reference = args.ref_file
                    p, k = get_corr(path_to_candidate, path_to_reference, args.ql_ranking_file, args.trec_eval_path,
                                    metric='ap')
                    res_p.setdefault(epoch, [])
                    res_p[epoch].append(p)
                    res_k.setdefault(epoch, [])
                    res_k[epoch].append(k)
            for i in res_k:
                p = np.mean(res_p[i])
                k = np.mean(res_k[i])
                record.write(f'epoch:{i}, pearson:{p}, kendall:{k}\n')



if __name__ == "__main__":
    main_inter()

