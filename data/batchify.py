# -*- coding: utf-8 -*-
# @Time   : 2023/02/06
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import torch
import math
import random
import numpy as np


class Batchify:
    def __init__(self, data, config, tag_num, shuffle=False):

        user, item, rating, reason, reason_label, video, video_label, interest, interest_label = [], [], [], [], [], [], [], [], []

        for x in data:
            pos_reason_list = eval(x['reason_tag'])
            pos_video_list = eval(x['video_tag'])
            pos_interest_list = eval(x['interest_tag'])

            a_pos_reason_tag = random.choice(pos_reason_list)
            a_pos_video_tag = random.choice(pos_video_list)
            a_pos_interest_tag = random.choice(pos_interest_list)

            # positive tag
            user.append(x['user_id'])
            item.append(x['video_id'])
            rating.append(x['rating'])
            reason.append(a_pos_reason_tag)
            video.append(a_pos_video_tag)
            interest.append(a_pos_interest_tag)
            reason_label.append(1.0)
            video_label.append(1.0)
            interest_label.append(1.0)

            # negative tag
            for _ in range(config['neg_sample_num']):
                user.append(x['user_id'])
                item.append(x['video_id'])
                rating.append(x['rating'])
                neg_ra = np.random.randint(tag_num)
                neg_vi = np.random.randint(tag_num)
                neg_in = np.random.randint(tag_num)
                while neg_ra in pos_reason_list:
                    neg_ra = np.random.randint(tag_num)
                reason.append(neg_ra)
                while neg_vi in pos_video_list:
                    neg_vi = np.random.randint(tag_num)
                video.append(neg_vi)
                while neg_in in pos_interest_list:
                    neg_in = np.random.randint(tag_num)
                interest.append(neg_in)
                reason_label.append(0.0)
                video_label.append(0.0)
                interest_label.append(0.0)

        self.user = torch.tensor(user, dtype=torch.int64)
        self.item = torch.tensor(item, dtype=torch.int64)
        self.rating = torch.tensor(rating, dtype=torch.float)
        self.reason = torch.tensor(reason, dtype=torch.int64)
        self.video = torch.tensor(video, dtype=torch.int64)
        self.interest = torch.tensor(interest, dtype=torch.int64)
        self.reason_label = torch.tensor(reason_label, dtype=torch.float)
        self.video_label = torch.tensor(video_label, dtype=torch.float)
        self.interest_label = torch.tensor(interest_label, dtype=torch.float)

        self.shuffle = shuffle
        self.batch_size = config['batch_size']
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        if self.shuffle:
            random.shuffle(self.index_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
        self.tag_num = tag_num

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        reason = self.reason[index]
        video = self.video[index]
        interest = self.interest[index]
        reason_label = self.reason_label[index]
        video_label = self.video_label[index]
        interest_label = self.interest_label[index]

        return user, item, rating, reason, reason_label, video, video_label, interest, interest_label


class TagTestBatchify:
    def __init__(self, data, config, tag_num, shuffle=False):
        self.tag_num = tag_num
        user, item, rating, pos_reason, pos_video, pos_interest, candi_reason, candi_video, candi_interest, = [], [], [], [], [], [], [], [], []
        for x in data:
            user.append(x['user_id'])
            item.append(x['video_id'])
            rating.append(x['rating'])
            pos_reason.append(eval(x['reason_tag']))
            pos_video.append(eval(x['video_tag']))
            pos_interest.append(eval(x['interest_tag']))
            candi_reason.append(self.get_candidate_tags(eval(x['reason_tag']), config['candidate_num']))
            candi_video.append(self.get_candidate_tags(eval(x['video_tag']), config['candidate_num']))
            candi_interest.append(self.get_candidate_tags(eval(x['interest_tag']), config['candidate_num']))

        self.user = torch.tensor(user, dtype=torch.int64)
        self.item = torch.tensor(item, dtype=torch.int64)
        self.rating = torch.tensor(rating, dtype=torch.float)
        self.candi_reason_tag = torch.tensor(candi_reason, dtype=torch.int64)
        self.candi_video_tag = torch.tensor(candi_video, dtype=torch.int64)
        self.candi_interest_tag = torch.tensor(candi_interest, dtype=torch.int64)

        self.shuffle = shuffle
        self.batch_size = config['batch_size']
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

        # Positive_tag acquisition, used to calculate sorting indicators, only used in the test phase
        self.positive_reason_tag = pos_reason
        self.positive_video_tag = pos_video
        self.positive_interest_tag = pos_interest

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        candi_reason_tag = self.candi_reason_tag[index]
        candi_video_tag = self.candi_video_tag[index]
        candi_interest_tag = self.candi_interest_tag[index]

        return user, item, candi_reason_tag, candi_video_tag, candi_interest_tag

    def get_candidate_tags(self, pos_tag_list, candidate_num):
        neg_tag_num = candidate_num - len(pos_tag_list)
        candi_tag_list = [i for i in pos_tag_list]
        for i in range(neg_tag_num):
            neg_tag = np.random.randint(self.tag_num)
            while neg_tag in pos_tag_list:
                neg_tag = np.random.randint(self.tag_num)
            candi_tag_list.append(neg_tag)
        random.shuffle(candi_tag_list)
        return candi_tag_list


class NegSamplingBatchify:
    r"""
    The function of negative sampling is provided for the label prediction task, and it is only used in the
    training phase.
    """

    def __init__(self, data, config, tag_num, shuffle=False):

        user, item, rating, reason_pos, reason_neg, video_pos, video_neg, interest_pos, interest_neg = [], [], [], [], [], [], [], [], []

        for x in data:
            pos_reason_list = eval(x['reason_tag'])
            pos_video_list = eval(x['video_tag'])
            pos_interest_list = eval(x['interest_tag'])

            a_pos_reason_tag = random.choice(pos_reason_list)
            a_pos_video_tag = random.choice(pos_video_list)
            a_pos_interest_tag = random.choice(pos_interest_list)

            for _ in range(config['neg_sample_num']):
                user.append(x['user_id'])
                item.append(x['video_id'])
                rating.append(x['rating'])
                reason_pos.append(a_pos_reason_tag)
                video_pos.append(a_pos_video_tag)
                interest_pos.append(a_pos_interest_tag)

                neg_ra = np.random.randint(tag_num)
                neg_vi = np.random.randint(tag_num)
                neg_in = np.random.randint(tag_num)
                while neg_ra in pos_reason_list:
                    neg_ra = np.random.randint(tag_num)
                reason_neg.append(neg_ra)
                while neg_vi in pos_video_list:
                    neg_vi = np.random.randint(tag_num)
                video_neg.append(neg_vi)
                while neg_in in pos_interest_list:
                    neg_in = np.random.randint(tag_num)
                interest_neg.append(neg_in)

        self.user = torch.tensor(user, dtype=torch.int64)
        self.item = torch.tensor(item, dtype=torch.int64)
        self.rating = torch.tensor(rating, dtype=torch.float)
        self.reason_pos = torch.tensor(reason_pos, dtype=torch.int64)
        self.reason_neg = torch.tensor(reason_neg, dtype=torch.int64)
        self.video_pos = torch.tensor(video_pos, dtype=torch.int64)
        self.video_neg = torch.tensor(video_neg, dtype=torch.int64)
        self.interest_pos = torch.tensor(interest_pos, dtype=torch.int64)
        self.interest_neg = torch.tensor(interest_neg, dtype=torch.int64)

        self.shuffle = shuffle
        self.batch_size = config['batch_size']
        self.sample_num = len(user)
        self.index_list = list(range(self.sample_num))
        if self.shuffle:
            random.shuffle(self.index_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        reason_pos = self.reason_pos[index]
        reason_neg = self.reason_neg[index]
        video_pos = self.reason_pos[index]
        video_neg = self.reason_neg[index]
        interest_pos = self.reason_pos[index]
        interest_neg = self.reason_neg[index]

        return user, item, rating, reason_pos, reason_neg, video_pos, video_neg, interest_pos, interest_neg

    def neg_tag_sampling(self):
        return


def sentence_format(sentence, max_len, pad, bos, eos):
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)


class ReviewBatchify:
    def __init__(self, data, word2idx, seq_len=15, batch_size=128, shuffle=False):
        bos = word2idx['<bos>']
        eos = word2idx['<eos>']
        pad = word2idx['<pad>']
        u, i, r, t = [], [], [], []
        for x in data:
            u.append(x['user_id'])
            i.append(x['video_id'])
            r.append(x['rating'])
            t.append(sentence_format(x['review'], seq_len, pad, bos, eos))

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        if self.shuffle:
            random.shuffle(self.index_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        return user, item, rating, seq
