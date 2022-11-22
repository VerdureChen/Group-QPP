from thop import profile
from thop import clever_format
from torchprofile import profile_macs
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
import random
from evaluation.test_trec_eval import validate
from model.cobert import COBERT
from model.ql_cobert import QL_COBERT
from model.bertbase import BERT
from get_corr import get_corr
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    small = np.ones_like(max)*1e-6
    return (data - min)/(max-min)

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

def do_eval(args, model, eval_dataloader, device, order_dict, score_dict):
    scores = []
    qids = []
    dids = []
    bias = []

    model.eval()
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t for t in batch_)
        # logger.info(f'batch:{len(batch_)}')
        with torch.no_grad():
            indexs = []
            input_ids, input_mask, segment_ids, query_ids, doc_ids, biass, label_ids = batch_
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            query_ids = query_ids.cpu().numpy()
            for idx in range(len(query_ids)):
                qid = str(query_ids[idx])
                if qid in order_dict:
                    index = order_dict[qid]
                else:
                    index = random.randint(0, len(order_dict) - 1)
                indexs.append(index)
                # did = doc_ids[idx]
            if args.model_name == 'poscobert':
                indexs = biass.long()
            indexs = torch.tensor(indexs, dtype=torch.long).to(device)
            ql_scores = []
            if args.model_name == 'ql_cobert':
                for idx in range(len(query_ids)):
                    qid = str(query_ids[idx])
                    did = str(doc_ids[idx])
                    ql_scores.append(score_dict[qid][did])
            ql_scores = torch.tensor(ql_scores).unsqueeze(1).to(device)
            # print(len(input_ids))
            # macs, params = profile(model, inputs=(input_ids, segment_ids, input_mask, label_ids, indexs,  args, ql_scores))
            # macs2, params2 = clever_format([macs, params], "%.3f")
            # print(macs, macs2)
            # print(params, params2)
            # break
            outputs = model(input_ids, segment_ids, input_mask, position_ids=indexs, args=args, ql_scores=ql_scores)
            logits = outputs
            if args.model_name == 'vanilla':
                probs = logits
            else:
                probs = logits

            scores.append(probs.detach().cpu().numpy())
            qids.append(query_ids[args.top_num:])
            bias.append(biass[args.top_num:])
            dids += doc_ids[args.top_num:]

    result = {}
    result['scores'] = np.concatenate(scores)
    # result['scores'] = scores
    result['qids'] = np.concatenate(qids)
    result['dids'] = dids
    result['biass'] = np.concatenate(bias)

    return result

def save_results(res, output_dir, model_dir, model_name='cobert'):
    query_psgs_ids = []
    scores = res['scores']
    max_dict = {}
    min_dict = {}
    for i in range(len(res['qids'])):
        qid = str(res['qids'][i])
        psg_id = str(res['dids'][i])
        pos_bias = str(res['biass'][i])
        query_psgs_ids.append([qid, psg_id, pos_bias])
        if qid not in max_dict:
            max_dict[qid]=float(scores[i])
            min_dict[qid]=float(scores[i])
        else:
            if float(scores[i]) > max_dict[qid]:
                max_dict[qid]=float(scores[i])
            elif float(scores[i]) < min_dict[qid]:
                min_dict[qid]=float(scores[i])

    # if model_name != 'poscobert':
    #     scores = []
    #     for i in range(len(res['qids'])):
    #         qid = str(res['qids'][i])
    #         min_s = min_dict[qid]
    #         max_s = max_dict[qid]
    #         score_now = res['scores'][i]
    #         score_now = (score_now - min_s)/(max_s - min_s)
    #         scores.append(score_now)

    print(len(scores))
    print(len(query_psgs_ids))
    assert len(scores) == len(query_psgs_ids)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rank1_result_path = os.path.join(output_dir,
                                         'results_rank1_{}_{}.txt'.format(
                                             model_dir, datetime.date.today().strftime('%Y_%m_%d')))
    with open(rank1_result_path, 'wt') as runfile:
        rank1_dict = {}
        for idx in range(len(scores)):
            qid = query_psgs_ids[idx][0]
            score = scores[idx]
            pos = query_psgs_ids[idx][2]
            if int(pos) == 1:
                rank1_dict[qid] = float(score)
        scores_rank1 = list(sorted(rank1_dict.items(), key=lambda x: (x[1], x[0]), reverse=True))
        for qid, score in scores_rank1:
            runfile.write(f'{qid} {score}\n')

    max_result_path = os.path.join(output_dir,
                                     'results_max_{}_{}.txt'.format(
                                         model_dir, datetime.date.today().strftime('%Y_%m_%d')))
    avg_result_path = os.path.join(output_dir,
                                   'results_avg_{}_{}.txt'.format(
                                       model_dir, datetime.date.today().strftime('%Y_%m_%d')))
    predictions_file_path = os.path.join(output_dir,
                                         'full_results_{}_{}.txt'.format(
                                          model_dir, datetime.date.today().strftime('%Y_%m_%d')))
    rerank_run = {}
    for idx in range(len(scores)):
        qid = query_psgs_ids[idx][0]
        psg_id = query_psgs_ids[idx][1]
        score = scores[idx]
        q_dict = rerank_run.setdefault(qid, {})
        if psg_id in q_dict.keys():
            if float(score) > q_dict[psg_id]:
                q_dict[psg_id] = float(score)
        else:
            q_dict[psg_id] = float(score)

    with open(predictions_file_path, 'wt') as runfile,\
         open(max_result_path, 'wt') as max_file,\
         open(avg_result_path, 'wt') as avg_file:
        max_dict = {}
        avg_dict = {}
        q_count = 0
        psg_count = 0
        for qid in rerank_run:
            scount = 0
            scores_run = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores_run):
                runfile.write(f'{qid} 0 {did} {i + 1} {score} run\n')
                psg_count = psg_count + 1
                if i == 0:
                    max_dict[qid] = float(score)
                scount += score
            avg = scount/len(scores_run)
            avg_dict[qid] = float(avg)
            q_count = q_count + 1
        scores_max = list(sorted(max_dict.items(), key=lambda x: (x[1], x[0]), reverse=True))
        for qid, score in scores_max:
            max_file.write(f'{qid} {score}\n')
        scores_avg = list(sorted(avg_dict.items(), key=lambda x: (x[1], x[0]), reverse=True))
        for qid, score in scores_avg:
            avg_file.write(f'{qid} {score}\n')
        print('total topic number:{}, total passage number:{}'.format(str(q_count), str(psg_count)))
    return rank1_result_path, max_result_path, avg_result_path

def get_metrics(result_path, args):
    logger.info(f'result_path:{result_path}')
    record_file = os.path.join(result_path, 'record_{}.txt'.format(datetime.date.today().strftime('%Y_%m_%d')))
    max_score = -1
    max_outfile = ''
    total_metrics = {}
    with open(record_file, 'w', encoding='utf-8') as record:
        dir_list = [item for item in os.listdir(result_path) if item.startswith('results_')]
        logger.info('*******')
        logger.info(dir_list)
        logger.info('*******')
        def num(ele):
            return int(ele.split('-')[-1].split('_')[0])
        dir_list.sort(key=num, reverse=True)

        for i, fil in enumerate(dir_list):
            res_file = os.path.join(result_path, fil)
            path_to_candidate = res_file
            path_to_reference = args.ref_file
            p, k = get_corr(path_to_candidate, path_to_reference, args.ql_ranking_file, args.trec_eval_path, metric='ap')
            mt = float(p)
            if mt > max_score:
                max_score = mt
                max_outfile = fil
            record.write('##########{}###########\n'.format(fil))
            record.write('pearson: {}\n'.format(p))
            record.write('kendall: {}\n'.format(k))
            record.write('#####################\n')
        record.write('MAX FILE:{}, MAX pearson:{}'.format(max_outfile, max_score))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        default=None,
                        type=str,
                        required=True,
                        help="The device you will run on.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The model for inference.")
    parser.add_argument("--encoder_model",
                        default=None,
                        type=str,
                        help="The encoder model dir.")
    parser.add_argument("--groupwise_model",
                        default=None,
                        type=str,
                        help="The groupwise model dir.")
    parser.add_argument("--attn_model",
                        default=None,
                        type=str,
                        help="The attn model dir.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="data dir")
    parser.add_argument("--data_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The csv data name.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="Which model to inference.")
    parser.add_argument("--modeldir_name",
                        default=None,
                        type=str,
                        required=True,
                        help="Which modeldir to inference.")
    parser.add_argument("--outdir_name",
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
    parser.add_argument("--label_type",
                        default=None,
                        type=str,
                        required=True,
                        help="label_type.")
    parser.add_argument("--metric",
                        default=None,
                        type=str,
                        required=True,
                        help="The metric.")
    parser.add_argument("--do_dev",
                        action='store_true',
                        help="Whether to run dev.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test.")
    parser.add_argument("--random",
                        action='store_true',
                        help="Whether to run test.")
    parser.add_argument("--multi_ckpts",
                        action='store_true',
                        help="To infer different steps' checkpoint.")
    parser.add_argument("--fold",
                        default=None,
                        type=int,
                        help="The number of folds in cross-validation.")
    parser.add_argument("--qid_split_dir",
                        default=None,
                        type=str,
                        help="The qid split file dir.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Batch size for eval.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task.")
    parser.add_argument("--top_num",
                        default=0,
                        type=int,
                        help="Total top doc number.")
    parser.add_argument("--rank_num",
                        default=0,
                        type=int,
                        help="Rank doc number.")
    parser.add_argument("--overlap",
                        default=0,
                        type=int,
                        help="Total overlap number.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions will be written.")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--ref_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The qrel file path.")
    parser.add_argument("--qpp_method",
                        default=None,
                        type=str,
                        required=True,
                        help="Clarity, ISD, NQC, QF, SD, SMV, UEF, WIG")
    parser.add_argument('--data_seed',
                        type=int,
                        default=2,
                        help="seed for data")
    parser.add_argument('--label_num',
                        type=int,
                        default=2,
                        help="label_num")

    # parser.add_argument("--num_eval_passages",
    #                     default=3000,
    #                     type=int,
    #                     help="The number of passages for a query")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    task_name = args.task_name
    if args.do_dev:
        tk_qpp = 'train'
        tk = 'dev'
    elif args.do_test:
        tk_qpp = 'test'
        tk = 'test'
    else:
        raise NotImplementedError

    label_list = get_labels(args.task_name.lower())
    num_labels = args.label_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    folds = list(range(args.fold))
    output_dir = os.path.join(args.output_dir, '{}', '{}/'.format(tk), args.outdir_name)
    for fold in folds:
        output_dir = output_dir.format(str(fold))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_dir = os.path.join(args.model_dir, str(fold), 'train', args.modeldir_name)
        model_dirs = []
        if args.multi_ckpts:
            dir_list = [item for item in os.listdir(model_dir) if item.startswith('checkpoint')]

            def num(ele):
                return int(ele.split('-')[-1])

            dir_list.sort(key=num, reverse=True)
            # dir_list.sort(key=num)
            logger.info('*******')
            logger.info(dir_list)
            logger.info('*******')
            for dir in dir_list:
                if dir.startswith('checkpoint'):
                    sub_model_dir = os.path.join(model_dir, dir)
                    model_dirs.append(sub_model_dir)
                    break
                else:
                    continue
        else:
            model_dirs.append(model_dir)

        if args.model_name == 'cobert':
            sampler = 'cross_cobert'
            if args.random:
                sampler = 'cross_random'
            # sampler = 'cross_random'
        elif args.model_name == 'ql_cobert':
            sampler = 'cross_cobert'
        elif args.model_name == 'vanilla':
            sampler = 'cross_random'
        elif args.model_name == 'poscobert':
            sampler = 'pos_cross_cobert'
        else:
            raise NotImplementedError

        for sub_model_dir in model_dirs:
            dir = sub_model_dir.split('/')[-1]
            if args.model_name == 'vanilla':
                model = BERT(sub_model_dir, num_labels=num_labels)
            elif args.model_name == 'cobert' or args.model_name == 'poscobert':
                model = COBERT(args.encoder_model, args.groupwise_model, args.attn_model, num_labels)
                model.load_state_dict(torch.load(os.path.join(sub_model_dir, 'weights.pt')), strict=False)
            elif args.model_name == 'ql_cobert':
                model = QL_COBERT(args.encoder_model, args.groupwise_model, args.attn_model, num_labels)
                model.load_state_dict(torch.load(os.path.join(sub_model_dir, 'weights.pt')))
            else:
                raise NotImplementedError

            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            qpp_file_path = os.path.join(args.output_dir, str(fold), '{}/'.format(tk_qpp), 'bsln', args.qpp_method)
            if args.do_dev:
                qpp = [item for item in os.listdir(qpp_file_path) if item.startswith('best')]
            elif args.do_test:
                qpp = [item for item in os.listdir(qpp_file_path) if not item.startswith('record')]
            assert len(qpp)==1
            qpp_file_path = os.path.join(qpp_file_path, qpp[0])
            order_dict = get_order_dict(qpp_file_path)
            score_dict = get_score_dict(args.ql_ranking_file)
            if args.do_dev:
                _, dataloader, _ = get_rank_task_dataloader([qpp_file_path], fold, task_name, 'dev', args, [sampler], args.eval_batch_size)
            elif args.do_test:
                _, dataloader, _ = get_rank_task_dataloader([qpp_file_path], fold, task_name, 'test', args, [sampler], args.eval_batch_size)
            else:
                raise NotImplementedError

            pred_res = do_eval(args, model, dataloader[0], device, order_dict, score_dict)
            if args.do_dev or args.do_test:
                save_results(pred_res, output_dir, dir, args.model_name)
                get_metrics(output_dir, args)
            else:
                raise NotImplementedError
        output_dir = os.path.join(args.output_dir, '{}', '{}/'.format(tk), args.outdir_name)
    if args.do_test:
        get_final_result(args.fold, args.final_path, args)


def get_final_result(fold, final_path, args):
    record_rank1_file = os.path.join(final_path,
                                     'record_rank1_{}_{}.txt'.format(args.outdir_name, datetime.date.today().strftime('%Y_%m_%d')))
    record_max_file = os.path.join(final_path,
                                     'record_max_{}_{}.txt'.format(args.outdir_name, datetime.date.today().strftime('%Y_%m_%d')))
    record_avg_file = os.path.join(final_path,
                                     'record_avg_{}_{}.txt'.format(args.outdir_name, datetime.date.today().strftime('%Y_%m_%d')))

    record_list = [record_rank1_file, record_max_file, record_avg_file]
    result_str = ['rank1', 'max', 'avg']
    for j, record_file in enumerate(record_list):
        res_p = {}
        res_k = {}
        with open(record_file, 'w', encoding='utf-8') as record:
            for i in range(fold):
                result_path = os.path.join(args.output_dir, '{}', 'test/', args.outdir_name)
                res_path = result_path.format(str(i))
                dir_list = [item for item in os.listdir(res_path) if item.startswith('results_{}'.format(result_str[j]))]
                ls = []
                lst = []
                for item in dir_list:
                    ck_num = item.split('-')[1].split('_')[0]
                    if ck_num in ls:
                        continue
                    else:
                        ls.append(ck_num)
                        lst.append(item)
                dir_list = lst
                logger.info('*******')
                logger.info(dir_list)
                logger.info('*******')

                def num(ele):
                    return int(ele.split('-')[-1].split('_')[0])

                dir_list.sort(key=num)

                for i, fil in enumerate(dir_list):
                    res_file = os.path.join(res_path, fil)
                    path_to_candidate = res_file
                    path_to_reference = args.ref_file
                    p, k = get_corr(path_to_candidate, path_to_reference, args.ql_ranking_file, args.trec_eval_path,
                                    metric='ap')
                    res_p.setdefault(i, [])
                    res_p[i].append(p)
                    res_k.setdefault(i, [])
                    res_k[i].append(k)
            for epo in res_k:
                p = np.mean(res_p[epo])
                k = np.mean(res_k[epo])
                record.write(f'epoch:{epo}, pearson:{p}, kendall:{k}\n')



if __name__ == "__main__":
    main()

