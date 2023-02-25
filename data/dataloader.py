# -*- coding: utf-8 -*-
# @Time   : 2023/02/06
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import pandas as pd
import os
import torch
import heapq


class TagDataLoader:

    def __init__(self, data_path, video_path, train_path, valid_path, test_path):
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.rating_scale = 5
        self.item_set = set()
        self.user_set = set()
        self.tag_num = 0
        self.interaction_num = 0
        self.initialize(data_path, video_path)
        self.user_num = len(self.user_set)
        self.item_num = len(self.item_set)
        self.trainData, self.trainset, self.validset, self.testset = self.load_data(train_path, valid_path, test_path)

        self.train_size = len(self.trainset)
        self.valid_size = len(self.validset)
        self.test_size = len(self.testset)

    def initialize(self, data_path, video_path):
        assert os.path.exists(data_path)
        reviews = pd.read_csv(data_path, header=0, usecols=[0, 1, 3, 4, 6, 7],
                              names=['user_id', 'video_id', 'reason_tag', 'rating', 'video_tag', 'interest_tag'],
                              sep='\t')
        reviews = reviews.to_dict('records')
        self.video_df = pd.read_csv(video_path, header=0, sep='\t')
        tag_list = [tag for video_row in self.video_df['tags'] for tag in eval(video_row)]
        self.tag_num = len(set(tag_list)) + 7
        for review in reviews:
            self.user_set.add(review['user_id'])
            self.item_set.add(review['video_id'])
            self.interaction_num += 1
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, train_path, valid_path, test_path):
        train_data = pd.read_csv(train_path, header=0, usecols=[0, 1, 3, 4, 6, 7],
                                 names=['user_id', 'video_id', 'reason_tag', 'rating', 'video_tag', 'interest_tag'],
                                 sep='\t')
        valid_data = pd.read_csv(valid_path, header=0, usecols=[0, 1, 3, 4, 6, 7],
                                 names=['user_id', 'video_id', 'reason_tag', 'rating', 'video_tag', 'interest_tag'],
                                 sep='\t')
        test_data = pd.read_csv(test_path, header=0, usecols=[0, 1, 3, 4, 6, 7],
                                names=['user_id', 'video_id', 'reason_tag', 'rating', 'video_tag', 'interest_tag'],
                                sep='\t')

        trainset = train_data.to_dict('records')
        validset = valid_data.to_dict('records')
        testset = test_data.to_dict('records')  # list [{'user_id':1, 'video_id':1, 'reason_tag':'[5, 2, 1,..]'}, {...}]

        return train_data, trainset, validset, testset

    def build_inter_matrix(self, model_name):
        r""" Bulid the user-tag & item-tag interaction matrix.

        Args:
            model_name: For which model

        Returns:
            interaction matrix

        """

        # X for user-side，Y for item-side
        X_r = torch.zeros((self.user_num, self.tag_num))  # user_reason_interaction
        Y_r = torch.zeros((self.item_num, self.tag_num))  # item_reason_interaction
        X_v = torch.zeros((self.user_num, self.tag_num))  # user_video_interaction
        Y_v = torch.zeros((self.item_num, self.tag_num))  # item_video_interaction
        X_i = torch.zeros((self.user_num, self.tag_num))  # user_interest_interaction
        Y_i = torch.zeros((self.item_num, self.tag_num))  # item_interest_interaction
        for d in self.trainset:
            u = d['user_id']
            i = d['video_id']
            for t in eval(d['reason_tag']):
                X_r[u][t] += 1
                Y_r[i][t] += 1
            for t in eval(d['video_tag']):
                X_v[u][t] += 1
                Y_v[i][t] += 1
            for t in eval(d['interest_tag']):
                X_i[u][t] += 1
                Y_i[i][t] += 1

        if model_name == 'EFM':
            X_r = self.efm_attention_score(X_r)
            Y_r = self.efm_quality_score(Y_r)
            X_v = self.efm_attention_score(X_v)
            Y_v = self.efm_quality_score(Y_v)
            X_i = self.efm_attention_score(X_i)
            Y_i = self.efm_quality_score(Y_i)
        elif model_name == 'AMF':
            X_r = self.amf_attention_score(X_r)
            Y_r = self.amf_quality_score(Y_r)
            X_v = self.amf_attention_score(X_v)
            Y_v = self.amf_quality_score(Y_v)
            X_i = self.amf_attention_score(X_i)
            Y_i = self.amf_quality_score(Y_i)
        return X_r, Y_r, X_v, Y_v, X_i, Y_i

    # EFM user-tag interaction matrix
    def efm_attention_score(self, tag_matrics):
        normal_tag_matrics = 1 + (self.rating_scale - 1) * (2 / (1 + torch.exp(-tag_matrics)) - 1)
        tag_matrics = torch.where(tag_matrics == 0, tag_matrics, normal_tag_matrics)
        return tag_matrics

    # EFM item-tag interaction matrix
    def efm_quality_score(self, tag_matrics):
        normal_tag_matrics = 1 + (self.rating_scale - 1) / (1 + torch.exp(-tag_matrics))
        tag_matrics = torch.where(tag_matrics == 0, tag_matrics, normal_tag_matrics)
        return tag_matrics

    # AMF user-tag interaction matrix
    def amf_attention_score(self, tag_matrics):
        # (user_num, 1)  Indicates the total number of interaction tags of the user
        sum_matrics = tag_matrics.sum(-1).view(-1, 1)
        importance = tag_matrics / sum_matrics
        normal_tag_matrics = importance * (self.rating_scale - 1) / (1 + torch.exp(-sum_matrics)) + 1
        return normal_tag_matrics

    # AMF item-tag interaction matrix
    def amf_quality_score(self, tag_matrics):
        # (item_num, 1)  Indicates the total number of interaction tags of the item
        sum_matrics = tag_matrics.sum(-1).view(-1, 1)
        importance = tag_matrics / (sum_matrics + 1e-7)

        # Method1 derectly using sentiment*inter_count
        normal_tag_matrics = importance / (1 + torch.exp(-sum_matrics)) * tag_matrics
        '''
        # Method2 using average_sentiment, and rescale to the range [1,5]. (similar to EFM)
        average_sentiment=1
        normal_tag_matrics = importance / (1 + torch.exp(-sum_matrics)) * average_sentiment * (self.rating_scale-1) +1
        '''
        return normal_tag_matrics

    def bulid_TriRank_matrix(self, tag_type):
        r"""
        Bulid the user-item & user-tag & item-tag interaction matrix for TriRank model
        Args:
            tag_type: "reason_tag" or "video_tag" or "interest_tag"

        Returns:
            tri-matrix

        """

        userItem = torch.zeros((self.user_num, self.item_num))
        userAspect = torch.zeros((self.user_num, self.tag_num))
        itemAspect = torch.zeros((self.item_num, self.tag_num))

        for d in self.trainset:
            u = d['user_id']
            i = d['video_id']
            r = d['rating']

            userItem[u][i] = r
            for t in eval(d[tag_type]):
                userAspect[u][t] += 1
                itemAspect[i][t] += 1

        # TF term weighting
        weight1 = torch.sum(userAspect, dim=1, keepdim=True)
        userAspect = userAspect / (weight1 + 1e-7)
        weight2 = torch.sum(itemAspect, dim=1, keepdim=True)
        itemAspect = itemAspect / (weight2 + 1e-7)

        return userItem, userAspect, itemAspect

    def build_history_interaction(self):
        # For DERM_H
        user_reason_list = []
        user_video_list = []
        user_interest_list = []
        for u in range(self.user_num):
            u_reason_list = []
            u_video_list = []
            u_interest_list = []
            u_data = self.trainData[self.trainData['user_id'] == u]
            for index, row in u_data.iterrows():
                u_reason_list.extend(eval(row['reason_tag']))
                u_video_list.extend(eval(row['video_tag']))
                u_interest_list.extend(eval(row['interest_tag']))
            user_reason_list.append(set(u_reason_list))
            user_video_list.append(set(u_video_list))
            user_interest_list.append(set(u_interest_list))

        item_tag_list = []
        for i in range(self.item_num):
            i_tag_list = []
            i_data = self.video_df[self.video_df['video_id'] == i]
            for index, row in i_data.iterrows():
                i_tag_list.extend(eval(row['tags']))
            item_tag_list.append(set(i_tag_list))

        return user_reason_list, user_video_list, user_interest_list, item_tag_list


class WordDictionary:  # word & feature
    def __init__(self):
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>']  # list
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}  # dict:{'<bos>':0, '<eos>':1, '<pad>':2, '<unk>':3}
        self.__word2count = {}

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):  # add word & record the word count
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20000):
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}


class EntityDictionary:  # user & item
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class ReviewDataLoader:

    def __init__(self, data_path, train_path, valid_path, test_path, vocab_size):
        self.word_dict = WordDictionary()
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx['<unk>']
        self.feature_set = set()
        self.user_num = len(self.user_dict)
        self.item_num = len(self.item_dict)
        self.interaction_num = 0
        self.trainset, self.validset, self.testset = self.load_data(train_path, valid_path, test_path)
        self.train_size = len(self.trainset)
        self.valid_size = len(self.validset)
        self.test_size = len(self.testset)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pd.read_csv(data_path, header=0, usecols=[0, 1, 4, 5],
                              names=['user_id', 'video_id', 'rating', 'review'], sep='\t')
        reviews = reviews.to_dict('records')
        for review in reviews:
            self.user_dict.add_entity(review['user_id'])
            self.item_dict.add_entity(review['video_id'])
            self.word_dict.add_sentence(review['review'])
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, train_path, valid_path, test_path):
        train_data = pd.read_csv(train_path, header=0, usecols=[0, 1, 4, 5],
                                 names=['user_id', 'video_id', 'rating', 'review'], sep='\t')
        valid_data = pd.read_csv(valid_path, header=0, usecols=[0, 1, 4, 5],
                                 names=['user_id', 'video_id', 'rating', 'review'], sep='\t')
        test_data = pd.read_csv(test_path, header=0, usecols=[0, 1, 4, 5],
                                names=['user_id', 'video_id', 'rating', 'review'], sep='\t')

        # word to id
        train_data['review'] = train_data['review'].apply(lambda x: self.seq2ids(x))
        valid_data['review'] = valid_data['review'].apply(lambda x: self.seq2ids(x))
        test_data['review'] = test_data['review'].apply(lambda x: self.seq2ids(x))

        train = train_data.to_dict('records')
        valid = valid_data.to_dict('records')
        test = test_data.to_dict('records')

        return train, valid, test

    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]
