
import os
import re
import sys
import subprocess
import argparse
from scipy import stats
from collections import OrderedDict
# parser = argparse.ArgumentParser()
# parser.add_argument("--trec_eval_parent_path",
#                         default=None,
#                         type=str,
#                         required=True,
#                         help="The student model dir.")
# args = parser.parse_args()
#
# parent_path = args.trec_eval_parent_path
# trec_eval_script_path = os.path.join(parent_path, 'trec_eval')
# sample_eval_script_path = os.path.join(parent_path, "sample_eval.pl")
# gd_eval_script_path = os.path.join(parent_path, "gdeval.pl")


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


def evaluate_trec(qrels, res, metrics):

  ''' all_trecs, '''
  command = [trec_eval_script_path, '-m', 'all_trec', '-M', '1000', qrels, res]
  output = run(command, get_ouput=True)
  output = str(output, encoding='utf-8')

  metrics_val = []
  for metric in metrics:
    metrics_val.append(re.findall(r'{0}\s+all.+\d+'.format(metric), output)[0].split('\t')[2].strip())

  # MAP = re.findall(r'map\s+all.+\d+', output)[0].split('\t')[2].strip()
  # P20 = re.findall(r'P_20\s+all.+\d+', output)[0].split('\t')[2].strip()

  return OrderedDict(zip(metrics, metrics_val))


def evaluate_sample_trec(qrels, res, metrics):
  command = [sample_eval_script_path, qrels, res]
  output = run(command, get_ouput=True)
  output = str(output, encoding='utf-8')

  metrics_val = []
  for metric in metrics:
    metrics_val.append(re.findall(r'{0}\s+all.+\d+'.format(metric), output)[0].split('\t')[4].strip())

  return OrderedDict(zip(metrics, metrics_val))


def evaluate_metrics(qrels, res, sample_qrels=None, metrics=None):
  normal_metrics = [met for met in metrics if not met.startswith('i')]
  infer_metrics = [met for met in metrics if met.startswith('i')]

  metrics_val_dict = OrderedDict()
  if len(normal_metrics) > 0:
    metrics_val_dict.update(evaluate_trec(qrels, res, metrics=normal_metrics))
  if len(infer_metrics) > 0:
    metrics_val_dict.update(evaluate_sample_trec(sample_qrels, res, metrics=infer_metrics))

  return metrics_val_dict

################################## perquery information ####################################
def evaluate_trec_perquery(qrels, res, metrics):

  ''' all_trecs, '''
  command = [trec_eval_script_path, '-m', 'all_trec', '-q', '-M', '1000', qrels, res]
  output = run(command, get_ouput=True)
  output = str(output, encoding='utf-8')

  metrics_val = []
  for metric in metrics:
    curr_res = re.findall(r'{0}\s+\t\d+.+\d+'.format(metric), output)
    curr_res = list(map(lambda x: float(x.split('\t')[-1]), curr_res))
    metrics_val.append(curr_res)

  return OrderedDict(zip(metrics, metrics_val))


def evaluate_sample_trec_perquery(qrels, res, metrics):
  command = [sample_eval_script_path, '-q', qrels, res]
  output = run(command, get_ouput=True)
  output = str(output, encoding='utf-8')

  metrics_val = []
  for metric in metrics:
    curr_res = re.findall(r'{0}\s+\t\d+.+\d+'.format(metric), output)
    curr_res = map(lambda x: float(x.split('\t')[-1]), curr_res)
    metrics_val.append(curr_res)

  return OrderedDict(zip(metrics, metrics_val))


def evaluate_metrics_perquery(qrels, res, sample_qrels=None, metrics=None):
  normal_metrics = [met for met in metrics if not met.startswith('i')]
  infer_metrics = [met for met in metrics if met.startswith('i')]

  metrics_val_dict = OrderedDict()
  if len(normal_metrics) > 0:
    metrics_val_dict.update(evaluate_trec_perquery(qrels, res, metrics=normal_metrics))
  if len(infer_metrics) > 0:
    metrics_val_dict.update(evaluate_sample_trec_perquery(sample_qrels, res, metrics=infer_metrics))

  return metrics_val_dict


def tt_test(qrels, res1, res2, sample_qrels=None, metrics=None):
  met_dict1 = evaluate_metrics_perquery(qrels, res1, sample_qrels, metrics)
  met_dict2 = evaluate_metrics_perquery(qrels, res2, sample_qrels, metrics)

  avg_met_dict1 = evaluate_metrics(qrels, res1, sample_qrels, metrics)
  avg_met_dict2 = evaluate_metrics(qrels, res2, sample_qrels, metrics)
  print(avg_met_dict1)
  print(avg_met_dict2)

  test_dict = OrderedDict()
  for met in met_dict1.keys():
    p_value = stats.ttest_rel(met_dict1.get(met), met_dict2.get(met))[1]
    test_dict.update({met: p_value})

  return test_dict

def evaluate_trec_per_query(qrels, res, tp, trec_eval_script_path):
  if tp == 'ndcg':
      tp = 'ndcg_cut_20'
  elif tp == 'ap':
      tp = 'map'
  command = [trec_eval_script_path, '-q', '-m', tp, qrels, res]
  output = run(command, get_ouput=True)
  output = str(output, encoding='utf-8')
  ndcg_lines = re.findall(tp+r'\s+\t\d+.+\d+', output)
  # print(re.findall(r'ndcg_cut_20\s+\tall+.+\d+', output)[0].split('\t')[2])
  ndcg10 = float(re.findall(tp+r'\s+\tall+.+\d+', output)[0].split('\t')[2])
  # print(ndcg10)
  # print(ndcg_lines)
  NDCG10_all = 0.
  NDCG10 = {}
  for line in ndcg_lines:
    tokens = line.split('\t')
    # print(tokens)
    assert tokens[0].strip() == tp
    qid, ndcg = tokens[1].strip(), float(tokens[2].strip())
    NDCG10[qid] = ndcg
    NDCG10_all += ndcg

  NDCG10_all /= len(NDCG10)
  # assert round(NDCG10_all, 4) == ndcg10, f'{NDCG10_all}-{ndcg10}'
  # print('ndcg@10: ', NDCG10_all)
  # print(len(NDCG10))
  # gd_command = [gd_eval_script_path, '-k', '20', qrels, res] #+ " | awk -F',' '{print $3}'"
  # gd_output = run(gd_command, get_ouput=True)
  # gd_output = str(gd_output, encoding='utf-8')
  # print(gd_output)

  # NDCG10_set, ERR10_set = [], []
  # for line in gd_output.split('\n')[1: -2]:
  #   ndcg, err = line.split(',')[2: 4]
  #   NDCG10_set.append(float(ndcg))
  #   ERR10_set.append(float(err))

  #print len(NDCG20_set)
  #print NDCG20_set

  return NDCG10, ndcg10

# def tt_test(qrels, res1, res2):
#   MAP_set1, P20_set1, NDCG20_set1 = evaluate_trec_per_query(qrels, res1)
#   MAP_set2, P20_set2, NDCG20_set2 = evaluate_trec_per_query(qrels, res2)
#   '''
#   print(P20_set1)
#   print(P20_set2)
#   print(NDCG20_set1)
#   print(NDCG20_set2)
#   print(len([t for t in np.asarray(MAP_set2) - np.asarray(MAP_set1) if t > 0]))
#   print(len([t for t in np.asarray(P20_set2) - np.asarray(P20_set1) if t > 0]))
#   print(len([t for t in np.asarray(NDCG20_set2) - np.asarray(NDCG20_set1) if t > 0]))
#   '''
#   t_value_map, p_value_map = stats.ttest_rel(MAP_set1, MAP_set2)
#   t_value_p20, p_value_p20 = stats.ttest_rel(P20_set1, P20_set2)
#   t_value_ndcg20, p_value_ndcg20 = stats.ttest_rel(NDCG20_set1, NDCG20_set2)
#
#   return p_value_map, p_value_p20, p_value_ndcg20


if __name__ == '__main__':
  # qrels = '/home/lcj/data/robust04/original/qrels.robust2004'
  # res = '/media/klaas/research/01_ir/BM25RocQEBase/robust_docnos.res'
  argv = sys.argv
  # res1, res2 = argv[1], argv[2]
  # print(tt_test(qrels, res1, res2))
  # print(evaluate_trec(argv[1], argv[2], ['map', 'P_10']))
  # print(evaluate_sample_trec(argv[3], argv[4], ['infNDCG', 'infAP']))
  #
  # print(evaluate_metrics(argv[1], argv[2], argv[3], ['map', 'P_10', 'infNDCG']))
  # print(evaluate_trec_perquery(argv[1], argv[2], ['Rprec', 'P_10']))
  # print(evaluate_sample_trec_perquery(argv[3], argv[4], ['infNDCG']))
  print(evaluate_trec_per_query(argv[1], argv[2]))
  # print(evaluate_metrics_perquery(argv[1], argv[2], argv[3], ['Rprec', 'P_10', 'infNDCG']))
  # print(tt_test(argv[1], argv[2], argv[3], argv[4], ['Rprec', 'P_10', 'infNDCG']))
  # print(evaluate_metrics(argv[1], argv[2], None, ['P_20', 'ndcg_cut_20', 'map_cut_100']))
  # print(tt_test(argv[1], argv[2], argv[3], None, ['map', 'ndcg_cut_20']))
