import os
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_final_result(fold_num, result_fold, output_path):
    folds = range(fold_num)
    if not os.path.exists(os.path.join(output_path)):
        os.makedirs(os.path.join(output_path))
    output_file = os.path.join(output_path,
                               'results_{}.txt'.format(result_fold.split('/')[-1]))
    with open(output_file, 'w', encoding='utf-8') as outf:
        count=0
        for fold in folds:
            result_path = result_fold.format(str(fold))
            dir_list = [item for item in os.listdir(result_path) if item.startswith('results')]
            logger.info('*******')
            logger.info(dir_list)
            logger.info('*******')
            assert len(dir_list)==1
            fil = dir_list[0]
            res_file = os.path.join(result_path, fil)
            with open(res_file, 'r', encoding='utf-8') as res:
                print(fil)
                for line in res:
                    outf.write(line)
                    count+=1
    print(f'total_line:{str(count)}')
    return output_file

def split_dl(result_file, dl_file1, dl_file2, output_path):
    dl19_qids = get_year_qids(dl_file1)
    dl20_qids = get_year_qids(dl_file2)
    dl19_dir = os.path.join(output_path, 'dl19')
    dl20_dir = os.path.join(output_path, 'dl20')
    dl19_path = os.path.join(output_path, 'dl19', 'results_{}'.format(result_file.split('/')[-1]))
    dl20_path = os.path.join(output_path, 'dl20', 'results_{}'.format(result_file.split('/')[-1]))
    with open(result_file, 'r', encoding='utf-8') as res:
        with open(dl19_path, 'w', encoding='utf-8') as dl19, \
             open(dl20_path, 'w', encoding='utf-8') as dl20:
            for line in res:
                qid = line.split(' ')[0]
                if qid in dl19_qids:
                    dl19.write(line)
                    continue
                if qid in dl20_qids:
                    dl20.write(line)
    return dl19_dir, dl20_dir



def get_year_qids(dl_file):
    qids = []
    with open(dl_file, 'r', encoding='utf-8') as dlf:
        dl = dlf.readlines()
        for line in dl:
            qids.append(str(line).strip())
    return qids