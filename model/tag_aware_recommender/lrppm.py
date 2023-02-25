# -*- coding: utf-8 -*-
# @Time   : 2023/02/10
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
LRPPM
################################################
Reference:
    Xu Chen et al. "Learning to Rank Features for Recommendation over Multiple Categories." in SIGIR 2016.
"""

import torch
import torch.nn as nn
from model.loss import BPRLoss


class LRPPM(nn.Module):
    r"""LRPPM is a tensor matrix factorization algorithm to Learn to Rank user Preferences based on Phrase-level
    sentiment analysis across Multiple categories.

    """

    def __init__(self, config):
        super(LRPPM, self).__init__()

        self.candidate_num = config['candidate_num']
        self.user_embeddings = nn.Embedding(config['user_num'], config['embedding_size'])
        self.item_embeddings = nn.Embedding(config['item_num'], config['embedding_size'])
        self.reason_tag_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])
        self.video_tag_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])
        self.interest_tag_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])

        self.bpr_loss = BPRLoss()
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=0.01)

    def forward(self):
        return

    def predict_rating(self, user, item):
        u_emb = self.user_embeddings(user)
        i_emb = self.item_embeddings(item)
        rating = torch.mul(u_emb, i_emb).sum(dim=1)
        return rating

    def predict_uit_score(self, user, item, tag, tag_type):
        # {reason:0, video:1, interest:3}
        u_emb = self.user_embeddings(user)
        i_emb = self.item_embeddings(item)
        if tag_type == 0:
            t_emb = self.reason_tag_embeddings(tag)
        elif tag_type == 1:
            t_emb = self.video_tag_embeddings(tag)
        else:
            t_emb = self.interest_tag_embeddings(tag)
        score = torch.mul(u_emb, t_emb).sum(dim=1) + \
                torch.mul(i_emb, t_emb).sum(dim=1) + \
                torch.mul(u_emb, i_emb).sum(dim=1)
        return score

    def calculate_rating_loss(self, user, item, rating_label):
        predicted_rating = self.predict_rating(user, item)
        rating_loss = self.mse_loss(predicted_rating, rating_label)
        return rating_loss

    def calculate_tag_loss(self, user, item, pos_tag, neg_tag, type):
        pos_score = self.predict_uit_score(user, item, pos_tag, type)
        neg_score = self.predict_uit_score(user, item, neg_tag, type)
        reason_loss = self.bpr_loss(pos_score, neg_score)
        return reason_loss

    def calculate_l2_loss(self):
        l2_loss = self.user_embeddings.weight.norm(2) + \
                  self.item_embeddings.weight.norm(2) + \
                  self.reason_tag_embeddings.weight.norm(2) + \
                  self.video_tag_embeddings.weight.norm(2) + \
                  self.interest_tag_embeddings.weight.norm(2)
        return l2_loss

    def rank_tags(self, user, item, tag, tag_type):
        u_emb = self.user_embeddings(user).unsqueeze(1)  # (B,E)
        i_emb = self.item_embeddings(item).unsqueeze(1)
        if tag_type == 0:
            t_emb = self.reason_tag_embeddings.weight[tag]  # (B,C,E)
        elif tag_type == 1:
            t_emb = self.video_tag_embeddings.weight[tag]
        else:
            t_emb = self.interest_tag_embeddings.weight[tag]
        t_emb = t_emb.transpose(1, 2)  # (B,C,E)->(B,E,C)
        uit_match_score = torch.bmm(u_emb, t_emb) + torch.bmm(i_emb, t_emb)
        return uit_match_score.squeeze(1)