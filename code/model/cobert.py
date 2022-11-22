from torch import nn
import argparse
import logging
import torch
import transformers
import sys
import os
import torch.nn.functional as F
from transformers import BertForTokenClassification, BertModel, BertConfig, BertForSequenceClassification



log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


class COBERT(torch.nn.Module):
    def __init__(self, pre_path_QT, pre_path_QB, pre_path_attn, num_labels, fold=None):
        super(COBERT, self).__init__()

        if fold is not None:
            dir_list = [item for item in os.listdir(pre_path_QT.format(str(fold))) if item.startswith('checkpoint')]
            if len(dir_list)!=0:
                pre_path_QT = os.path.join(pre_path_QT.format(str(fold)), dir_list[0])

        self.QT_BERT = BertForSequenceClassification.from_pretrained(pre_path_QT, num_labels=num_labels)
        self.ATTN_BERT = BertForSequenceClassification.from_pretrained(pre_path_attn, num_labels=num_labels)
        # config1 = BertConfig(num_hidden_layers=2, num_labels=1, max_position_embeddings=1000)
        self.QB_BERT = BertForTokenClassification.from_pretrained(pre_path_QB, num_labels=1)
        # self.QB_BERT = BertForTokenClassification(config1)
        self.dropout = nn.Dropout(0.1)
        logger.info('Initializing co-bert successfully!')


    def init_weights(self):
        nn.init.xavier_normal_(self.classfier.weight.data)
        nn.init.xavier_normal_(self.dense.weight.data)
        self.classfier.bias.data.zero_()
        self.dense.bias.data.zero_()


    def forward(self, QT_input_ids, QT_token_type_ids=None, QT_attention_masks=None, logit_labels=None, position_ids=None, args=None, ql_scores=None):
        batch_size = QT_input_ids.size(0)

        QT_output = self.QT_BERT(input_ids=QT_input_ids, token_type_ids=QT_token_type_ids, output_hidden_states=True,
                                                   attention_mask=QT_attention_masks, return_dict=True)
        QT_logits = QT_output.logits
        QT_pooled_output = QT_output.hidden_states[-1][:,0]
        device = QT_input_ids.device

        # QT_attn = QT_pooled_output[top_num:].unsqueeze(1)
        # ATTN_input = ''
        # for i in range(top_num):
        #     attn_start = QT_pooled_output[i:i+1].unsqueeze(0)
        #     attn_start = attn_start.expand(batch_size-top_num, 1, -1)
        #     if i==0:
        #         ATTN_input = torch.cat((attn_start, QT_attn), 1)
        #     else:
        #         now = torch.cat((attn_start, QT_attn), 1)
        #         ATTN_input = torch.cat((ATTN_input, now), 0)
        # if top_num == 0:
        #     ATTN_input = QT_pooled_output[top_num:].unsqueeze(1)
        # ATTN_output = self.ATTN_BERT(inputs_embeds=ATTN_input, output_hidden_states=True,
        #                              return_dict=True)
        # ATTN_now = ''
        # logits_for_soft = QT_logits[:top_num, 1].expand(batch_size-top_num, top_num)
        # probs = F.softmax(logits_for_soft, dim=-1)
        # # logger.info(f'size attn:{str(QT_logits[:4, 1].shape)}')
        # for i in range(top_num):
        #     if i==0:
        #         ATTN_now = ATTN_output.hidden_states[-1][:batch_size-top_num, -1]*probs[:,i].unsqueeze(1)
        #     else:
        #         now = ATTN_output.hidden_states[-1][(batch_size-top_num)*i:(batch_size-top_num)*(i+1), -1]\
        #               *probs[:,i].unsqueeze(1)
        #         ATTN_now = ATTN_now+now
        # if top_num == 0:
        #     ATTN_now = ATTN_output.hidden_states[-1][:batch_size-top_num, -1]
        # ATTN_rep = (ATTN_now + QT_pooled_output[top_num:]) / 2
        QB_input =QT_pooled_output.unsqueeze(0)
        QB_token_type_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        # if overlap!=0:
        #     QB_token_type_ids[0:overlap] = 1
        if logit_labels is not None:
            # logit_labels = logit_labels[top_num:].unsqueeze(0)
            QB_output = self.QB_BERT(inputs_embeds=QB_input, token_type_ids=QB_token_type_ids, position_ids=position_ids,
                                     return_dict=True)
            # return QB_output.logits.squeeze(0)[:,1]
            return QB_output.logits.squeeze(0)
        else:
            QB_output = self.QB_BERT(inputs_embeds=QB_input, token_type_ids=QB_token_type_ids, position_ids=position_ids,
                                     return_dict=True)
            # return QB_output.logits.squeeze(0)[:,1]
            return QB_output.logits.squeeze(0)
