import subprocess
import gc


def run(command, get_ouput=False):
  try:
    if get_ouput:
      process = subprocess.Popen(command, stdout=subprocess.PIPE)
      output, err = process.communicate()
      exit_code = process.wait()
      return output
    else:
      subprocess.call(command)
  except subprocess.CalledProcessError as e:
    print(e)


def trec_eval(qrelf, runf, metric1, metric2, metric3, metric4):
    trec_eval_f = '/home1/cxy/COBERT/code/bin/trec_eval'
    # gc.collect()
    # output = subprocess.check_output([trec_eval_f, '-m', metric1, '-m', metric2, '-m', metric3,'-m', metric4, qrelf, runf], close_fds=True).decode().rstrip()
    # output = subprocess.check_output(
    #     [trec_eval_f, '-m', 'all_trec', qrelf, runf]).decode().rstrip()
    command = [trec_eval_f, '-m', metric1, '-m', metric2, '-m', metric3,'-m', metric4, qrelf, runf]
    output = run(command, get_ouput=True)
    output = str(output, encoding='utf-8')
    output = output.replace('\t', ' ').split('\n')
    # assert len(output) == 1
    # print(output)
    score_dict={}
    for item in output:
        if item is not '':
            # print(item.split(' '))
            score_dict[item.split(' ')[0]] = item.split(' ')[-1]
    return score_dict

def validate(qrelf, runf, ndcg='ndcg_cut_10'):
    VALIDATION_METRIC1 = 'P.20'
    VALIDATION_METRIC2 = 'map_cut.100'
    VALIDATION_METRIC3 = 'map'
    if ndcg == 'ndcg_cut_10':
        ndcg='ndcg_cut.10'
    else:
        ndcg='ndcg_cut.20'
    VALIDATION_METRIC4 = ndcg
    return trec_eval(qrelf, runf, VALIDATION_METRIC1, VALIDATION_METRIC2,VALIDATION_METRIC3, VALIDATION_METRIC4)


if __name__ == '__main__':
    qrelf = r'/data/cxy/msmarco_passage/2019qrels-pass.txt'
    runf = r'/data/cxy/MS/output/ms_passage/test/dl19_bertbase_50000/results_checkpoint-200000_2021_08_20.txt'
    for i in range(1,2):
        score = validate(qrelf,runf.format(str(i)))
        # print(f'fold {i}')
        for key, val in score.items():
            print(f"{i}:{key}\t{val}")
        print('\n')