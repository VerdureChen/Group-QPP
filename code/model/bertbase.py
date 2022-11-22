from torch import nn
import argparse
import logging
import torch
import sys
import transformers
from transformers import BertForSequenceClassification

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

class BERT(torch.nn.Module):
    def __init__(self, ckpt_path, num_labels):
        super(BERT, self).__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained(ckpt_path,num_labels=num_labels)
        self.label_num = num_labels
        logger.info('Initializing Bert successfully!')

    def forward(self,QT_input_ids, QT_token_type_ids=None, QT_attention_masks=None, logit_labels=None, position_ids=None, args=None, ql_scores=None):
        Bert_output = self.bert_model(input_ids=QT_input_ids, token_type_ids=QT_token_type_ids,
                                 attention_mask=QT_attention_masks, return_dict=True)

        if self.label_num == 2:
            return Bert_output.logits[:,1]
        else:
            return Bert_output.logits


    def save_pretrained(self,outpath):
        self.bert_model.save_pretrained(outpath)

