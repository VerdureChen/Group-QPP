import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import logging
import sys
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class COBERTSampler(Sampler):
    def __init__(self, data, batchsize, args, total_top=0):
        self.data = data
        self.batchsize = batchsize
        self.args = args
        self.total_top = total_top
        self.batchnum_wo_drop, self.batchnum_wi_drop, self.indices = self.get_indices()
        logger.info('COBERTSampler start successfully!')

    def __iter__(self):
        _, _, indices = self.get_indices()
        return iter(indices)

    def get_indices(self):
        topnum = self.args.top_num
        overlap = self.args.overlap
        indices = []
        olen = self.batchsize - topnum
        q_ids = np.unique(self.data.query_ids)
        for qid in q_ids:
            qindex = torch.tensor(np.where(self.data.query_ids == qid)[0])
            topindex = qindex[:topnum]
            s = 0+self.total_top
            e = olen+self.total_top
            while e < len(qindex):
                part = qindex[s:e]
                unit = torch.cat([topindex, part], dim=0)
                indices.append(unit)
                s = e - overlap
                e = s + olen
            e = len(qindex)
            indices.append(torch.cat([topindex, qindex[s:e]], dim=0))
        random.shuffle(indices)
        batchnum_wo_drop = len(indices)
        batchnum_wi_drop = len([i for i in indices if len(i) == self.batchsize])
        indices = torch.cat(indices, dim=0)
        indices = indices
        return batchnum_wo_drop, batchnum_wi_drop, indices

    def __len__(self):
        return len(self.indices)

class cross_posCOBERTSampler(Sampler):
    def __init__(self, data, batchsize, args, qids, qpp_file_path, total_top=0):
        self.data = data
        self.batchsize = batchsize
        self.args = args
        self.qids = qids
        self.rank_num = args.rank_num
        self.total_top = total_top
        self.qpp_order = self.get_initial_qpp_list(qpp_file_path)
        print(self.qpp_order)
        self.batchnum_wo_drop, self.batchnum_wi_drop, self.indices = self.get_indices()
        logger.info('cross_COBERTSampler start successfully!')

    def __iter__(self):
        _, _, indices = self.get_indices()
        return iter(indices)

    def get_initial_qpp_list(self, qpp_file_path):
        qpp_id_order = []
        qpp_dict = {}
        with open(qpp_file_path, 'r', encoding='utf-8') as qpp_file:
            for line in qpp_file:
                qid, score = line.strip().split(' ')
                qpp_dict[qid] = float(score)
        order = sorted(qpp_dict.items(),key=lambda item:item[1], reverse=True)
        for item in order:
            qpp_id_order.append(item[0])
        for q in self.qids:
            if q not in qpp_id_order:
                i = random.randint(0, len(qpp_id_order) - 1)
                qpp_id_order.insert(i, q)
        q_return = []
        for q in qpp_id_order:
            if q in self.qids:
                q_return.append(q)
        return q_return


    def get_indices(self):
        topnum = self.args.top_num
        overlap = self.args.overlap
        olen = self.batchsize - topnum
        # q_ids = np.unique(self.data.query_ids)
        indices = []
        for qid in self.qpp_order:
            num_indices = []
            for num in range(1, self.rank_num):
                qid = int(qid)
                try:
                    qindex = torch.tensor(np.where((self.data.biass == num)&(self.data.query_ids == qid))[0])
                    assert len(qindex) == 1, len(qindex)
                except:
                    continue
                num_indices.append(qindex)
            num_indices = torch.cat(num_indices, dim=0)
            topindex = num_indices[:topnum]
            s = 0 + self.total_top
            e = olen + self.total_top
            while e < len(num_indices):
                part = num_indices[s:e]
                unit = torch.cat([topindex, part], dim=0)
                indices.append(unit)
                s = e - overlap
                e = s + olen
            e = len(num_indices)
            indices.append(torch.cat([topindex, num_indices[s:e]], dim=0))
        random.shuffle(indices)
        batchnum_wo_drop = len(indices)
        batchnum_wi_drop = len([i for i in indices if len(i) == self.batchsize])
        indices = torch.cat(indices, dim=0)
        indices = indices
        return batchnum_wo_drop, batchnum_wi_drop, indices

    def __len__(self):
        return len(self.indices)


class cross_COBERTSampler(Sampler):
    def __init__(self, data, batchsize, args, qids, qpp_file_path, total_top=0):
        self.data = data
        self.batchsize = batchsize
        self.args = args
        self.qids = qids
        self.rank_num = args.rank_num
        # self.total_top = total_top
        self.qpp_order = self.get_initial_qpp_list(qpp_file_path)
        print(self.qpp_order)
        self.batchnum_wo_drop, self.batchnum_wi_drop, self.indices = self.get_indices()
        logger.info('cross_COBERTSampler start successfully!')

    def __iter__(self):
        _, _, indices = self.get_indices()
        return iter(indices)

    def get_initial_qpp_list(self, qpp_file_path):
        qpp_id_order = []
        qpp_dict = {}
        with open(qpp_file_path, 'r', encoding='utf-8') as qpp_file:
            for line in qpp_file:
                qid, score = line.strip().split(' ')
                qpp_dict[qid] = float(score)
        order = sorted(qpp_dict.items(),key=lambda item:item[1], reverse=True)
        for item in order:
            qpp_id_order.append(item[0])
        for q in self.qids:
            if q not in qpp_id_order:
                i = random.randint(0, len(qpp_id_order) - 1)
                qpp_id_order.insert(i, q)
        q_return = []
        for q in qpp_id_order:
            if q in self.qids:
                q_return.append(q)
        return q_return


    def get_indices(self):
        # topnum = self.args.top_num
        # overlap = self.args.overlap
        # olen = self.batchsize - topnum
        # q_ids = np.unique(self.data.query_ids)
        indices = []
        for num in range(1, self.rank_num):
            num_indices = []
            for qid in self.qpp_order:
                qid = int(qid)
                try:
                    qindex = torch.tensor(np.where((self.data.biass == num) & (self.data.query_ids == qid))[0])
                    assert len(qindex) == 1, len(qindex)
                except:
                    continue
                num_indices.append(qindex)
            num_indices = torch.cat(num_indices, dim=0)
            s = 0
            e = self.batchsize
            while e < len(num_indices):
                part = num_indices[s:e]
                indices.append(part)
                s = e
                e = e + self.batchsize
            e = len(num_indices)
            indices.append(num_indices[s:e])
        random.shuffle(indices)
        batchnum_wo_drop = len(indices)
        batchnum_wi_drop = len([i for i in indices if len(i) == self.batchsize])
        indices = torch.cat(indices, dim=0)
        indices = indices
        return batchnum_wo_drop, batchnum_wi_drop, indices

    def __len__(self):
        return len(self.indices)


class cross_RandomSampler(Sampler):
    def __init__(self, data, qids, args):
        self.data = data
        self.qids = qids
        self.args = args
        self.rank_num = args.rank_num
        self.indices = self.get_indices()
        logger.info('cross_RandomSampler start successfully!')

    def __iter__(self):
        indices = self.get_indices()
        return iter(indices)

    def get_indices(self):
        indices = []
        q_ids = np.unique(self.data.query_ids)
        for qid in q_ids:
            if str(qid) not in self.qids:
                continue
            for num in range(1, self.rank_num):
                qid = int(qid)
                try:
                    qindex = torch.tensor(np.where((self.data.biass == num) & (self.data.query_ids == qid))[0])
                    assert len(qindex) == 1
                except:
                    continue

                indices.append(qindex)
        indices = torch.cat(indices, dim=0)
        indices = torch.tensor([indices[i] for i in torch.randperm(len(indices))])
        return indices

    def __len__(self):
        return len(self.indices)


# class cross_RandomSampler(Sampler):
#     def __init__(self, data, qids):
#         self.data = data
#         self.qids = qids
#         self.indices = self.get_indices()
#         logger.info('cross_RandomSampler start successfully!')
#
#     def __iter__(self):
#         indices = self.get_indices()
#         return iter(indices)
#
#     def get_indices(self):
#         indices = []
#         q_ids = np.unique(self.data.query_ids)
#         for qid in q_ids:
#             if str(qid) not in self.qids:
#                 continue
#             qindex = torch.tensor(np.where(self.data.query_ids == qid)[0])
#             indices.append(qindex)
#         indices = torch.cat(indices, dim=0)
#         indices = torch.tensor([indices[i] for i in torch.randperm(len(indices))])
#         return indices
#
#     def __len__(self):
#         return len(self.indices)

class cross_negrand_COBERTSampler(Sampler):
    def __init__(self, data, batchsize, args, qids, qpp_file_path, total_top=0):
        self.data = data
        self.batchsize = batchsize
        self.args = args
        self.qids = qids
        self.rank_num = args.rank_num
        # self.total_top = total_top
        self.qpp_order = self.get_initial_qpp_list(qpp_file_path)
        print(self.qpp_order)
        self.batchnum_wo_drop, self.batchnum_wi_drop, self.indices = self.get_indices()
        logger.info('cross_COBERTSampler start successfully!')

    def __iter__(self):
        _, _, indices = self.get_indices()
        return iter(indices)

    def get_initial_qpp_list(self, qpp_file_path):
        qpp_id_order = []
        qpp_dict = {}
        with open(qpp_file_path, 'r', encoding='utf-8') as qpp_file:
            for line in qpp_file:
                qid, score = line.strip().split(' ')
                qpp_dict[qid] = float(score)
        order = sorted(qpp_dict.items(),key=lambda item:item[1], reverse=True)
        for item in order:
            qpp_id_order.append(item[0])
        for q in self.qids:
            if q not in qpp_id_order:
                i = random.randint(0, len(qpp_id_order) - 1)
                qpp_id_order.insert(i, q)
        q_return = []
        for q in qpp_id_order:
            if q in self.qids:
                q_return.append(q)
        return q_return


    def get_indices(self):
        # topnum = self.args.top_num
        # overlap = self.args.overlap
        # olen = self.batchsize - topnum
        # q_ids = np.unique(self.data.query_ids)
        indices = []
        for num in range(1, self.rank_num):
            num_indices = []
            for qid in self.qpp_order:
                qid = int(qid)
                try:
                    qindex = torch.tensor(np.where((self.data.biass == num) & (self.data.query_ids == qid))[0])
                    assert len(qindex) == 1, len(qindex)
                except:
                    continue
                num_indices.append(qindex)
            num_indices = torch.cat(num_indices, dim=0)
            s = 0
            e = self.batchsize
            while e < len(num_indices):
                part = num_indices[s:e]
                indices.append(part)
                s = e
                e = e + self.batchsize
            e = len(num_indices)
            indices.append(num_indices[s:e])
        random.shuffle(indices)
        batchnum_wo_drop = len(indices)
        batchnum_wi_drop = len([i for i in indices if len(i) == self.batchsize])
        indices = torch.cat(indices, dim=0)
        indices = indices
        return batchnum_wo_drop, batchnum_wi_drop, indices

    def __len__(self):
        return len(self.indices)



class COBERTBatchSampler:
    def __init__(self, sampler, batchsize, drop_last=False):
        self.sampler = sampler
        self.batch_size = batchsize
        self.drop_last = drop_last
        logger.info('COBERTBatchSampler start successfully!')

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                    i < len(sampler_list) - 1
                    and self.sampler.data.biass[idx]
                    != self.sampler.data.biass[sampler_list[i + 1]]
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.sampler.batchnum_wi_drop
        else:
            return self.sampler.batchnum_wo_drop


class posCOBERTBatchSampler:
    def __init__(self, sampler, batchsize, drop_last=False):
        self.sampler = sampler
        self.batch_size = batchsize
        self.drop_last = drop_last
        logger.info('COBERTBatchSampler start successfully!')

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                    i < len(sampler_list) - 1
                    and self.sampler.data.query_ids[idx]
                    != self.sampler.data.query_ids[sampler_list[i + 1]]
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.sampler.batchnum_wi_drop
        else:
            return self.sampler.batchnum_wo_drop