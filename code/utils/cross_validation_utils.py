import random
import os

def split_qids_to_fold(repeat_num, fold_num, qid_path, out_path, qrel_file):
    '''

    :param repeat_num:
    :param fold_num:
    :param qid_path:
    :param out_path:
    :return:
    split/repeat_num/0:for train
    split/repeat_num/1:for test
    '''
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for t in range(repeat_num):
        out_p = os.path.join(out_path, str(t))
        if not os.path.exists(out_p):
            os.makedirs(out_p)
        qids = []
        qrel_qids=[]
        with open(qid_path, 'r', encoding='utf-8') as qid_f,\
             open(qrel_file, 'r', encoding='utf-8') as qrel:
            for line in qrel:
                topic = line.split(' ')[0]
                if topic not in qrel_qids:
                    qrel_qids.append(topic)
            q_lines = qid_f.readlines()
            for line in q_lines:
                qid = line.strip()
                if qid in qrel_qids:
                    qids.append(qid)
            print(qids)
        random.shuffle(qids)
        print(qids)
        assert len(qids) >= fold_num, "To many folds!"
        base_num = len(qids)//fold_num
        leave_num = len(qids)%fold_num
        s = 0
        e = base_num
        splits = []
        for i in range(fold_num):
            split = qids[s:e]
            splits.append(split)
            s = e
            e += base_num
        if leave_num != 0:
            lt = random.sample(range(fold_num), leave_num)
            for k in lt:
                splits[k].append(qids[s])
                s += 1
        assert s == len(qids), "s is not equal to the qid number!"
        for item in splits:
            print(item)
            print('\n')
        for i in range(fold_num):
            outf = os.path.join(out_p, str(i))
            with open(outf, 'w', encoding='utf-8') as outfile:
                for item in splits[i]:
                    outfile.write(f'{str(item)}\n')


def get_fold_qids(fold, split_path, task='train'):
    test_split = [1]
    train_split = [0]
    print(f'test:{test_split}')
    print(f'train:{train_split}')
    qids = []
    if task == 'train':
        qlist = train_split
    elif task == 'test':
        qlist = test_split
    elif task == 'dev':
        qlist = train_split
    else:
        raise NotImplementedError
    for i in qlist:
        q_path = os.path.join(split_path, str(fold), str(i))
        with open(q_path, 'r', encoding='utf-8') as qf:
            q_lines = qf.readlines()
            for line in q_lines:
                qids.append(line.strip())

    return qids