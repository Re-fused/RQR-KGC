#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

from curriculum import adaptive_hard_count_cap

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode,
                 relation_pool_negative_ratio=0.0,
                 curriculum_hard_negative_min_ratio=0.0,
                 curriculum_hard_negative_max_ratio=0.0):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.relation_pool_negative_ratio = float(
            relation_pool_negative_ratio
        )
        if not 0.0 <= self.relation_pool_negative_ratio <= 1.0:
            raise ValueError(
                'relation_pool_negative_ratio must be between 0 and 1'
            )
        self.curriculum_hard_negative_min_ratio = float(
            curriculum_hard_negative_min_ratio
        )
        self.curriculum_hard_negative_max_ratio = float(
            curriculum_hard_negative_max_ratio
        )
        if not (0.0 <= self.curriculum_hard_negative_min_ratio
                <= self.curriculum_hard_negative_max_ratio <= 1.0):
            raise ValueError(
                'curriculum hard-negative ratios must satisfy '
                '0 <= min <= max <= 1'
            )
        if (self.relation_pool_negative_ratio > 0.0
                and self.curriculum_hard_negative_max_ratio > 0.0):
            raise ValueError(
                'static and curriculum relation-pool sampling are exclusive'
            )
        self.curriculum_hard_negative_size = int(np.ceil(
            self.negative_sample_size
            * self.curriculum_hard_negative_max_ratio
        ))
        self._relation_pool_rng = None
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        if (self.relation_pool_negative_ratio > 0.0
                or self.curriculum_hard_negative_max_ratio > 0.0):
            self.relation_head, self.relation_tail = (
                self.get_relation_head_and_tail(self.triples)
            )
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        if self.mode == 'head-batch':
            true_entities = self.true_head[(relation, tail)]
            relation_pool = (
                self.relation_head[relation]
                if (self.relation_pool_negative_ratio > 0.0
                    or self.curriculum_hard_negative_max_ratio > 0.0)
                else None
            )
        elif self.mode == 'tail-batch':
            true_entities = self.true_tail[(head, relation)]
            relation_pool = (
                self.relation_tail[relation]
                if (self.relation_pool_negative_ratio > 0.0
                    or self.curriculum_hard_negative_max_ratio > 0.0)
                else None
            )
        else:
            raise ValueError('Training batch mode %s not supported' % self.mode)

        if self.curriculum_hard_negative_max_ratio > 0.0:
            uniform_negative = self._sample_filtered(
                None, true_entities, self.negative_sample_size
            )
            relation_pool_rng = self._get_relation_pool_rng()
            hard_negative = self._sample_filtered(
                relation_pool, true_entities,
                self.curriculum_hard_negative_size,
                max_rounds=8, rng=relation_pool_rng
            )
            hard_available = hard_negative.size
            if hard_available < self.curriculum_hard_negative_size:
                hard_negative = np.concatenate([
                    hard_negative,
                    self._sample_filtered(
                        None, true_entities,
                        self.curriculum_hard_negative_size - hard_available,
                        rng=relation_pool_rng
                    )
                ])

            hard_count_cap = adaptive_hard_count_cap(
                relation_pool.size,
                self.nentity,
                self.negative_sample_size,
                self.curriculum_hard_negative_min_ratio,
                self.curriculum_hard_negative_max_ratio,
                hard_available,
            )
            return (
                torch.LongTensor(positive_sample),
                torch.LongTensor(uniform_negative),
                torch.LongTensor(hard_negative),
                torch.tensor(hard_count_cap, dtype=torch.long),
                subsampling_weight,
                self.mode
            )

        relation_pool_size = int(round(
            self.negative_sample_size
            * self.relation_pool_negative_ratio
        ))
        relation_negative = self._sample_filtered(
            relation_pool, true_entities, relation_pool_size,
            max_rounds=8
        )
        uniform_size = (
            self.negative_sample_size - relation_negative.size
        )
        uniform_negative = self._sample_filtered(
            None, true_entities, uniform_size
        )
        negative_sample = np.concatenate([
            relation_negative, uniform_negative
        ])

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode

    def _get_relation_pool_rng(self):
        if self._relation_pool_rng is None:
            mode_offset = 0x6A09E667 if self.mode == 'head-batch' else 0xBB67AE85
            seed = (torch.initial_seed() + mode_offset) % (2 ** 32)
            self._relation_pool_rng = np.random.RandomState(seed)
        return self._relation_pool_rng

    def _sample_filtered(
            self, pool, true_entities, size, max_rounds=None, rng=None):
        if size <= 0:
            return np.empty(0, dtype=np.int64)

        rng = np.random if rng is None else rng

        samples = []
        sample_size = 0
        rounds = 0
        while sample_size < size:
            if pool is None:
                # Preserve the original uniform-sampling RNG stream when the
                # relation-pool ratio is zero.
                draw_size = self.negative_sample_size * 2
                candidates = rng.randint(
                    self.nentity, size=draw_size
                )
            else:
                draw_size = max(2 * (size - sample_size), 16)
                candidates = pool[rng.randint(
                    pool.size, size=draw_size
                )]
            candidates = candidates[np.in1d(
                candidates, true_entities,
                assume_unique=True, invert=True
            )]
            samples.append(candidates)
            sample_size += candidates.size
            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break

        if not samples:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(samples)[:size]
    
    @staticmethod
    def collate_fn(data):
        if len(data[0]) == 6:
            positive_sample = torch.stack([_[0] for _ in data], dim=0)
            uniform_negative = torch.stack([_[1] for _ in data], dim=0)
            hard_negative = torch.stack([_[2] for _ in data], dim=0)
            hard_count_cap = torch.stack([_[3] for _ in data], dim=0)
            subsample_weight = torch.cat([_[4] for _ in data], dim=0)
            mode = data[0][5]
            return (
                positive_sample, uniform_negative, hard_negative,
                hard_count_cap, subsample_weight, mode
            )
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    @staticmethod
    def get_relation_head_and_tail(triples):
        relation_head = {}
        relation_tail = {}
        for head, relation, tail in triples:
            relation_head.setdefault(relation, set()).add(head)
            relation_tail.setdefault(relation, set()).add(tail)
        relation_head = {
            relation: np.asarray(sorted(entities), dtype=np.int64)
            for relation, entities in relation_head.items()
        }
        relation_tail = {
            relation: np.asarray(sorted(entities), dtype=np.int64)
            for relation, entities in relation_tail.items()
        }
        return relation_head, relation_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
