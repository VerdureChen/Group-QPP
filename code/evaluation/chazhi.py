import os
from test_trec_eval import validate
import numpy as np
from tqdm import tqdm
import logging
import datetime
import sys
import shutil
import math
# qb_path = '/data/chenxiaoyang/MS/data/origin/output/robust/{}/test/result-fulltop1qb-2020-11-03/results_robust_2020_11_08.txt'
# qt_path = '/data/chenxiaoyang/MS/data/origin/output/robust/{}/test/result-fulltop1qt-2020-11-03/results_robust_2020_11_08.txt'

output_dir = '/data/chenxiaoyang/MS/data/origin/output2/robust/{}/insert/dev/result-0128-bertbaseMS-3e6-multiattn-com/{}_{}'
test_output_dir = output_dir.replace('dev', 'test')

# output_dir_qt_qb = '/data/chenxiaoyang/MS/data/origin/output/robust/{}/insert/dev/result-fulltop1-2020-11-03/qt_qb'
# output_dir_pre_qt_qb = '/data/chenxiaoyang/MS/data/origin/output/robust/{}/insert/dev/result-fulltop1-2020-11-03/pre_qt_qb'
# output_dir_pre_qtb = '/data/chenxiaoyang/MS/data/origin/output/robust/{}/insert/dev/result-fulltop1-2020-11-03/pre_qtb'
# pre_qt_qb_out_file = '/data/chenxiaoyang/MS/data/origin/output/robust/{}/insert/dev/result-fulltop1-2020-11-03/pre_qt_qb/results_lam{}_lamp{}.txt'
# pre_qtb_out_file = '/data/chenxiaoyang/MS/data/origin/output/robust/{}/insert/dev/result-fulltop1-2020-11-03/pre_qtb/results_lam{}.txt'
# qt_qb_out_file = '/data/chenxiaoyang/MS/data/origin/output/robust/{}/insert/dev/result-fulltop1-2020-11-03/qt_qb/results_lam{}.txt'
# qrel_file = '/data/chenxiaoyang/MS/data/origin/gov2/qrels'
# pre_path = '/data/chenxiaoyang/MS/data/origin/gov2/gov2_title_DPH.res'
qrel_file = '/home1/cxy/COBERT/data/ms/2020dl/2020-qrels-pass-final.txt'
pre_path = '/data/chenxiaoyang/MS/data/origin/output/robust/MS_ranking.txt'
# five_fold_result_path = '/data/chenxiaoyang/MS/data/origin/output2/robust/metrics/insert/test_{}_{}_insert_{}.txt'

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    small = np.ones_like(max)*1e-6
    return (data+small - min)/(max-min)

def read_top_score(qb_dict = None):
    top_dict = {}
    top_score = []
    if qb_dict is None:
        with open(pre_path, 'r', encoding='utf8') as top100_f:
            for line in top100_f:
                qid, _, docid, rank, score, _ = line.strip().split(' ')
                if qid not in top_dict:
                    top_dict[qid] = {}
                top_dict[qid][docid] = float(score)
                top_score.append(float(score))
            top_score = np.array(top_score)
            # print(top_score)
            # print(type(top_score))
            # scal = minmaxscaler(top_score)
            scal = top_score
            scal = scal.tolist()
        with open(pre_path, 'r', encoding='utf8') as top100_f:
            for line, sc_score in zip(top100_f, scal):
                qid, _, docid, rank, score, _ = line.strip().split(' ')
                top_dict[qid][docid] = float(sc_score)
            # for k, v in top_dict.items():
            #     print(f'{k}:{v}')
            logger.info(f'finish construct top_dict, has {len(top_dict)} queries.')
            return top_dict
    else:
        with open(pre_path, 'r', encoding='utf8') as top100_f:
            for line in top100_f:
                qid, _, docid, rank, score, _ = line.strip().split(' ')
                if qid not in qb_dict:
                    continue
                if qid not in top_dict:
                    top_dict[qid] = {}
                top_dict[qid][docid] = float(score)
                top_score.append(float(score))
            top_score = np.array(top_score)
            # print(top_score)
            # print(type(top_score))
            scal = minmaxscaler(top_score)
            scal = scal.tolist()
        with open(pre_path, 'r', encoding='utf8') as top100_f:
            for line, sc_score in zip(top100_f, scal):
                qid, _, docid, rank, score, _ = line.strip().split(' ')
                if qid not in qb_dict:
                    continue
                top_dict[qid][docid] = float(sc_score)
            # for k, v in top_dict.items():
            #     print(f'{k}:{v}')
            logger.info(f'finish construct top_dict, has {len(top_dict)} queries.')
            return top_dict

def pre_qt_qb_insert(fold, mode='full'):

    lams = [float(item) / 10 for item in range(0, 11, 1)]
    lams_pre = [float(item) / 10 for item in range(0, 11, 1)]
    task_name = 'pre_qt_qb'
    out_path = output_dir.format(fold, task_name, mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'results_lam{}_lamp{}.txt')
    for lam in lams:
        for lam_p in lams_pre:
            with open(out_file.format(str(lam), str(lam_p)), 'w', encoding='utf-8') as out, \
                    open(qb_path.format(str(fold)), 'r', encoding='utf-8') as qb, \
                    open(qt_path.format(str(fold)), 'r', encoding='utf-8') as qt:
                qb_dict = {}
                for qbs in qb:
                    qb_topid, _, qb_docid, _, qb_score, _ = qbs.strip().split(' ')
                    if qb_topid not in qb_dict:
                        qb_dict[qb_topid] = {}
                    qb_dict[qb_topid][qb_docid] = float(qb_score)
                qt_dict = {}
                for qts in qt:
                    qt_topid, _, qt_docid, _, qt_score, _ = qts.strip().split(' ')
                    if qt_topid not in qt_dict:
                        qt_dict[qt_topid] = {}
                    qt_dict[qt_topid][qt_docid] = float(qt_score)
                if mode == 'full':
                    top_dict = read_top_score()
                elif mode == 'fold_sp':
                    top_dict = read_top_score(qb_dict)
                final_dict = {}
                for topid, doc_dict in qb_dict.items():
                    if topid not in final_dict:
                        final_dict[topid] = {}
                    for docid, score in qb_dict[topid].items():
                        final_dict[topid][docid] = (1 - lam_p) * (
                                    lam * qt_dict[topid][docid] + (1 - lam) * qb_dict[topid][docid]) + lam_p * \
                                                   top_dict[topid][docid]
                for qid in final_dict:
                    scores = list(sorted(final_dict[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
                    for i, (did, score) in enumerate(scores):
                        out.write(f'{qid} 0 {did} {i + 1} {score} run\n')

def pre_qtb_insert(fold, mode='full'):
    qtb_path = '/data/chenxiaoyang/MS/data/origin/output2/robust/{}/dev/result-0128-bertbaseMS-3e6-multiattn-com'
    test_path = qtb_path.replace('dev','test')
    file_list = [item for item in os.listdir(test_path.format(str(fold))) if item.startswith('results')]
    assert len(file_list)==1
    file_name = file_list[0].split('_')[1]

    lams = [float(item) / 10 for item in range(0, 11, 1)]
    task_name = 'qtb'
    out_path = output_dir.format(fold, task_name, mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'results_{}_lam{}.txt')
    qtb_path = qtb_path.format(str(fold))
    dir_list = [item for item in os.listdir(qtb_path) if item.startswith('results') if file_name in item]
    if len(dir_list) == 2:
        dir_list = [dir_list[0]]
    assert len(dir_list)==1
    if mode == 'full':
        top_dict = read_top_score()
    for dir in dir_list:
        qtb_file = os.path.join(qtb_path, dir)
        file_name = dir.split('_')[1]
        for lam in lams:
            scores = []
            qb_dict = {}
            with open(out_file.format(str(file_name),str(lam)), 'w', encoding='utf-8') as out, \
                    open(qtb_file, 'r', encoding='utf-8') as qb:
                for qbs in qb:
                    qb_topid, _, qb_docid, _, qb_score, _ = qbs.strip().split(' ')
                    if qb_topid not in qb_dict:
                        qb_dict[qb_topid] = {}
                    qb_dict[qb_topid][qb_docid] = math.log(float(qb_score))
            #         scores.append(float(qb_score))
            # scal = minmaxscaler(scores)
            # scal = scal.tolist()
            # with open(out_file.format(str(file_name), str(lam)), 'w', encoding='utf-8') as out, \
            #         open(qtb_file, 'r', encoding='utf-8') as qb:
            #     for qbs,s in zip(qb,scal):
            #         qb_topid, _, qb_docid, _, qb_score, _ = qbs.strip().split(' ')
            #         if qb_topid not in qb_dict:
            #             qb_dict[qb_topid] = {}
                    # print(s)
                    # qb_dict[qb_topid][qb_docid] = math.log(float(s))
                if mode == 'fold_sp':
                    top_dict = read_top_score(qb_dict)
                final_dict = {}
                for topid, doc_dict in qb_dict.items():
                    if topid not in final_dict:
                        final_dict[topid] = {}
                    for docid, score in qb_dict[topid].items():
                        final_dict[topid][docid] = lam * qb_dict[topid][docid] + (1 - lam) * top_dict[topid][docid]
                for qid in final_dict:
                    scores = list(sorted(final_dict[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
                    for i, (did, score) in enumerate(scores):
                        out.write(f'{qid} 0 {did} {i + 1} {score} run\n')

    return test_path


def qt_qb_insert(fold, mode='full'):
    lams = [float(item) / 10 for item in range(0, 11, 1)]
    task_name = 'qt_qb'
    out_path = output_dir.format(fold, task_name, mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'results_lam{}.txt')
    for lam in lams:
        with open(out_file.format(str(lam)), 'w', encoding='utf-8') as out,\
             open(qb_path.format(str(fold)), 'r', encoding='utf-8') as qb,\
              open(qt_path.format(str(fold)), 'r', encoding='utf-8') as qt:
            qb_dict = {}
            for qbs in qb:
                qb_topid, _,  qb_docid, _, qb_score, _= qbs.strip().split(' ')
                if qb_topid not in qb_dict:
                    qb_dict[qb_topid]={}
                qb_dict[qb_topid][qb_docid] = float(qb_score)
            qt_dict = {}
            for qts in qt:
                qt_topid, _,  qt_docid, _, qt_score, _= qts.strip().split(' ')
                if qt_topid not in qt_dict:
                    qt_dict[qt_topid]={}
                qt_dict[qt_topid][qt_docid] = float(qt_score)
            final_dict={}
            for topid, doc_dict in qb_dict.items():
                if topid not in final_dict:
                    final_dict[topid]={}
                for docid, score in qb_dict[topid].items():
                    final_dict[topid][docid] = lam*qt_dict[topid][docid]+(1-lam)*qb_dict[topid][docid]
            for qid in final_dict:
                    scores = list(sorted(final_dict[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
                    for i, (did, score) in enumerate(scores):
                        out.write(f'{qid} 0 {did} {i + 1} {score} run\n')
            print(f'finish lamda:{lam}')



def count_insert(fold, test_path, lam_out, mode='full'):

    file_list = [item for item in os.listdir(test_path.format(str(fold))) if item.startswith('results')]
    assert len(file_list)==1


    lams = [float(item) / 10 for item in range(0, 11, 1) if str(float(item) / 10) in lam_out]
    task_name = 'qtb'
    out_path = test_output_dir.format(fold, task_name, mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'results_lam{}.txt')
    qtb_path = test_path.format(str(fold))

    qtb_file = os.path.join(qtb_path, file_list[0])
    if mode == 'full':
        top_dict = read_top_score()
    for lam in lams:
        qb_dict = {}
        scores = []
        with open(out_file.format(str(lam)), 'w', encoding='utf-8') as out, \
                open(qtb_file, 'r', encoding='utf-8') as qb:
            for qbs in qb:
                qb_topid, _, qb_docid, _, qb_score, _ = qbs.strip().split(' ')
                if qb_topid not in qb_dict:
                    qb_dict[qb_topid] = {}
                qb_dict[qb_topid][qb_docid] = float(qb_score)
                scores.append(float(qb_score))
        scal = minmaxscaler(scores)
        scal = scal.tolist()
        with open(out_file.format(str(lam)), 'w', encoding='utf-8') as out, \
                open(qtb_file, 'r', encoding='utf-8') as qb:
            for qbs, s in zip(qb, scal):
                qb_topid, _, qb_docid, _, qb_score, _ = qbs.strip().split(' ')
                if qb_topid not in qb_dict:
                    qb_dict[qb_topid] = {}
                qb_dict[qb_topid][qb_docid] = math.log(float(s))
            if mode == 'fold_sp':
                top_dict = read_top_score(qb_dict)
            final_dict = {}
            for topid, doc_dict in qb_dict.items():
                if topid not in final_dict:
                    final_dict[topid] = {}
                for docid, score in qb_dict[topid].items():
                    final_dict[topid][docid] = lam * qb_dict[topid][docid] + (1 - lam) * top_dict[topid][docid]
            for qid in final_dict:
                scores = list(sorted(final_dict[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
                for i, (did, score) in enumerate(scores):
                    out.write(f'{qid} 0 {did} {i + 1} {score} run\n')


def two_file_inpo(out_file, qtb_file, colbert_file):
    lams = [float(item) / 10 for item in range(0, 11, 1)]
    for lam in lams:
        scores = []
        qb_dict = {}
        top_dict = {}
        with open(out_file.format(str(lam)), 'w', encoding='utf-8') as out, \
                open(qtb_file, 'r', encoding='utf-8') as qb, \
                open(colbert_file, 'r', encoding='utf-8') as colbert:
            for qbs in qb:
                qb_topid, _, qb_docid, _, qb_score, _ = qbs.strip().split(' ')
                if qb_topid not in qb_dict:
                    qb_dict[qb_topid] = {}
                qb_dict[qb_topid][qb_docid] = math.log(float(qb_score))
            for col in colbert:
                qb_topid, _, qb_docid, _, qb_score, _ = col.strip().split(' ')
                if qb_topid not in top_dict:
                    top_dict[qb_topid] = {}
                top_dict[qb_topid][qb_docid] = math.log(float(qb_score))

            final_dict = {}
            for topid, doc_dict in qb_dict.items():
                if topid not in final_dict:
                    final_dict[topid] = {}
                for docid, score in qb_dict[topid].items():
                    try:
                        final_dict[topid][docid] = lam * qb_dict[topid][docid] + (1 - lam) * top_dict[topid][docid]
                    except:
                        pass
            for qid in final_dict:
                scores = list(sorted(final_dict[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
                for i, (did, score) in enumerate(scores):
                    out.write(f'{qid} 0 {did} {i + 1} {score} run\n')


def compute_metrics(output_dir):
    dev_record_file = os.path.join(output_dir, 'record_{}.txt'.format(datetime.date.today().strftime('%Y_%m_%d')))
    max_score = -1
    max_outfile = ''
    result_path = output_dir
    r_metric = {}
    with open(dev_record_file, 'w', encoding='utf-8') as dev_record:
        dir_list = [item for item in os.listdir(result_path) if item.startswith('result')]

        def num(ele):
            return float(ele.split('-')[-1].split('_')[1].split('.')[0]+ele.split('-')[-1].split('_')[1].split('.')[1])

        try:
            dir_list.sort(key=num, reverse=True)
        except:
            pass
        # logger.info('*******')
        print(dir_list)
        # logger.info('*******')

        for fil in dir_list:
            res_file = os.path.join(result_path, fil)
            path_to_candidate = res_file
            path_to_reference = qrel_file
            metrics = validate(path_to_reference, path_to_candidate)
            r_metric[fil] = metrics
            p20 = float(metrics['map'])
            if p20 > max_score:
                max_score = p20
                max_outfile = fil
            dev_record.write('##########{}###########\n'.format(fil))
            for metric in sorted(metrics):
                dev_record.write('{}: {}\n'.format(metric, metrics[metric]))
            dev_record.write('#####################\n')
        dev_record.write('MAX FILE:{}, MAX map:{}'.format(max_outfile, str(max_score)))
        lamb = max_outfile
        return lamb

if __name__ == "__main__":
    out_file = r'/home1/cxy/COBERT/output/ms_passage/inpo/result_{}'
    output_dir = r'//home1/cxy/COBERT/output/ms_passage/inpo'
    qtb_file = r'/home1/cxy/COBERT/output/ms_passage/test/dl20_cobert_lst_2data/results_checkpoint-50000_2021_09_02.txt'
    colbert_file = r'/home1/cxy/COBERT/output/ms_passage/test/dl20_cobert_lst_2data_randomsam/results_checkpoint-50000_2021_09_02.txt'
    two_file_inpo(out_file, qtb_file, colbert_file)
    compute_metrics(output_dir)
    # mode = ['full', 'fold_sp', 'query_sp']
    # task_name = ['qt_qb', 'pre_qt_qb', 'qtb']
    # mod = mode[0]
    # task = task_name[2]
    # folds = range(1, 6)
    # f_metric = {}
    # for fold in folds:
    #     # ckpt_path = r'/data/chenxiaoyang/MS/data/origin/output2/robust/{}/train/ckpt-list-0107-all-10epoch_bertbaseMS-3e6bsln-64'
    #     # test_path = r'/data/chenxiaoyang/MS/data/origin/output2/robust/{}/test/result-listwise-20201-01-07-10epoch_bertbaseMS-seq-order3e6-64-ins'
    #     # qt_qb_insert(fold)
    #     # if task == 'qt_qb':
    #     #     qt_qb_insert(fold)
    #     # elif task == 'pre_qtb':
    #     #     pre_qtb_insert(fold, mod)
    #     # elif task == 'pre_qt_qb':
    #     #     pre_qt_qb_insert(fold, mod)
    #
    #     test_path = pre_qtb_insert(fold, mod)
    #     lam_out = compute_metrics(fold, task, mod)
    #     # ckpt_path = ckpt_path.format(str(fold))
    #     # test_path = test_path.format(str(fold))
    #     # if not os.path.exists(os.path.join(ckpt_path, 'best4insert')):
    #     #     os.makedirs(os.path.join(ckpt_path, 'best4insert'))
    #     # best_dir = os.path.join(ckpt_path, 'best4insert')
    #     # check_name = lam_out.split('_')[1]
    #     # check_path = os.path.join(ckpt_path, check_name)
    #     # dir_list = [item for item in os.listdir(best_dir) if item.startswith('checkpoint')]
    #     # best_dir = os.path.join(best_dir, dir_list[0])
    #     # lam_fil = os.path.join(best_dir, 'lam.txt')
    #     # shutil.copytree(check_path, best_dir)
    #     # with open(lam_fil, 'w', encoding='utf-8') as lamf:
    #     #     lamf.write(lam_out)
    #     # lam_out = ''
    #     # with open(lam_fil, 'r', encoding='utf-8') as lamf:
    #     #     for line in lamf:
    #     #         lam_out = line
    #     count_insert(fold, test_path, lam_out, mod)
    #     # if fold == 1:
    #     #     f_metric = r_metric
    #     # else:
    #     #     for fil in r_metric:
    #     #         metric = f_metric[fil]
    #     #         for p in metric:
    #     #             f_metric[fil][p] = float(f_metric[fil][p]) + float(r_metric[fil][p])
    #     print(f'finish count fold{fold}')
    #
    # # for fil in f_metric:
    # #     for p in f_metric[fil]:
    # #         f_metric[fil][p] = f_metric[fil][p]/5
    # # five_fold_result_path=five_fold_result_path.format(task, mod, datetime.date.today().strftime('%Y_%m_%d'))
    # # if not os.path.exists(five_fold_result_path):
    # #     os.makedirs(five_fold_result_path)
    # # with open(five_fold_result_path, 'w', encoding='utf-8') as result:
    # #     print('sorting....')
    # #     print(f_metric)
    # #     unsorted = f_metric
    # #     for fil in f_metric:
    # #         result.write('\n\n')
    # #         result.write(fil)
    # #         result.write('\n')
    # #         # print(type(unsorted[fil]),unsorted[fil])
    # #         for metric in sorted(unsorted[fil]):
    # #             result.write('{}: {}\n'.format(metric, unsorted[fil][metric]))
    # #
    # #     for m in ['P_20', 'map', 'map_cut_100', 'ndcg_cut_20']:
    # #         unsorted = f_metric
    # #         result.write('**********************{}******************************\n'.format(m))
    # #         sorted_p20 = sorted(unsorted.items(),key=lambda x:x[1][m], reverse=True)
    # #         for fil in sorted_p20:
    # #             # print(sorted_p20)
    # #             result.write('\n\n')
    # #             result.write(fil[0])
    # #             result.write('\n')
    # #             # print(type(sorted_p20[0]), sorted_p20[0])
    # #             for metric in sorted(fil[1]):
    # #                 result.write('{}: {}\n'.format(metric, fil[1][metric]))
    print('finished!')