# -*- coding: utf-8 -*-
# @Time   : 2023/02/08
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
EFM
################################################
Reference:
    Yongfeng Zhang et al. "Explicit Factor Models for Explainable Recommendation based on Phrase-level Sentiment Analysis." in SIGIR 2014.
"""

import torch
import torch.nn as nn


class EFM(nn.Module):
    r"""EFM incorporates both user-tag and item-tag relations as well as user-item ratings into a new unified
    hybrid matrix factorization framework

    """

    def __init__(self, config):
        super(EFM, self).__init__()

        self.user_embeddings = nn.Embedding(config['user_num'], config['embedding_size'])
        self.item_embeddings = nn.Embedding(config['item_num'], config['embedding_size'])
        self.user_h_embeddings = nn.Embedding(config['user_num'], config['embedding_size'])
        self.item_h_embeddings = nn.Embedding(config['item_num'], config['embedding_size'])
        self.reason_tag_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])
        self.video_tag_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])
        self.interest_tag_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])

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
        u_h_emb = self.user_h_embeddings(user)
        i_h_emb = self.item_h_embeddings(item)
        complete_u_emb = torch.cat((u_emb, u_h_emb), 1)
        complete_i_emb = torch.cat((i_emb, i_h_emb), 1)
        rating = torch.mul(complete_u_emb, complete_i_emb).sum(dim=1)
        return rating

    def predict_tag_score(self, x_emb, tag_emb):
        tag_score = torch.mm(x_emb, torch.transpose(tag_emb, 0, 1))
        return tag_score

    def calculate_rating_loss(self, user, item, rating_label):
        predicted_rating = self.predict_rating(user, item)
        rating_loss = self.mse_loss(predicted_rating, rating_label)
        return rating_loss

    def calculate_reason_loss(self, user, item, user_reason_label, item_reason_label):
        u_emb = self.user_embeddings(user)
        i_emb = self.item_embeddings(item)
        t_emb = self.reason_tag_embeddings.weight
        ut_score = self.predict_tag_score(u_emb, t_emb)
        it_score = self.predict_tag_score(i_emb, t_emb)
        reason_loss = self.mse_loss(ut_score, user_reason_label) + self.mse_loss(it_score, item_reason_label)
        return reason_loss

    def calculate_video_loss(self, user, item, user_video_label, item_video_label):
        u_emb = self.user_embeddings(user)
        i_emb = self.item_embeddings(item)
        t_emb = self.reason_tag_embeddings.weight
        ut_score = self.predict_tag_score(u_emb, t_emb)
        it_score = self.predict_tag_score(i_emb, t_emb)
        video_loss = self.mse_loss(ut_score, user_video_label) + self.mse_loss(it_score, item_video_label)
        return video_loss

    def calculate_interest_loss(self, user, item, user_interest_label, item_interest_label):
        u_emb = self.user_embeddings(user)
        i_emb = self.item_embeddings(item)
        t_emb = self.reason_tag_embeddings.weight
        ut_score = self.predict_tag_score(u_emb, t_emb)
        it_score = self.predict_tag_score(i_emb, t_emb)
        interest_loss = self.mse_loss(ut_score, user_interest_label) + self.mse_loss(it_score, item_interest_label)
        return interest_loss

    def calculate_non_negative_reg(self):
        u_reg = torch.sum((torch.abs(self.user_embeddings.weight) - self.user_embeddings.weight))
        i_reg = torch.sum((torch.abs(self.item_embeddings.weight) - self.item_embeddings.weight))
        u_h_reg = torch.sum((torch.abs(self.user_h_embeddings.weight) - self.user_h_embeddings.weight))
        i_h_reg = torch.sum((torch.abs(self.item_h_embeddings.weight) - self.item_h_embeddings.weight))
        reason_tag_reg = torch.sum((torch.abs(self.reason_tag_embeddings.weight) - self.reason_tag_embeddings.weight))
        video_tag_reg = torch.sum((torch.abs(self.video_tag_embeddings.weight) - self.video_tag_embeddings.weight))
        interest_tag_reg = torch.sum((torch.abs(self.interest_tag_embeddings.weight) - self.interest_tag_embeddings.weight))
        non_negative_reg = u_reg + i_reg + u_h_reg + i_h_reg + reason_tag_reg + video_tag_reg + interest_tag_reg
        return non_negative_reg

    def calculate_l2_loss(self):
        l2_loss = self.user_embeddings.weight.norm(2) + \
                  self.item_embeddings.weight.norm(2) + \
                  self.user_h_embeddings.weight.norm(2) + \
                  self.item_h_embeddings.weight.norm(2) + \
                  self.reason_tag_embeddings.weight.norm(2) + \
                  self.video_tag_embeddings.weight.norm(2) + \
                  self.interest_tag_embeddings.weight.norm(2)
        return l2_loss

    def predict_reason_score(self, user, item, tag):
        u_emb = self.user_embeddings(user).unsqueeze(1)  # (B,E)->(B,1,E)
        i_emb = self.item_embeddings(item).unsqueeze(1)
        t_emb = self.reason_tag_embeddings.weight[tag].transpose(1, 2)  # (B,C,E)->(B,E,C)
        u_score = torch.bmm(u_emb, t_emb).squeeze(1)  # (B,1,C)->(B,C)
        i_score = torch.bmm(i_emb, t_emb).squeeze(1)
        score = torch.mul(u_score, i_score)
        return score

    def predict_video_score(self, user, item, tag):
        u_emb = self.user_embeddings(user).unsqueeze(1)  # (B,E)->(B,1,E)
        i_emb = self.item_embeddings(item).unsqueeze(1)
        t_emb = self.video_tag_embeddings.weight[tag].transpose(1, 2)  # (B,C,E)->(B,E,C)
        u_score = torch.bmm(u_emb, t_emb).squeeze(1)  # (B,1,C)->(B,C)
        i_score = torch.bmm(i_emb, t_emb).squeeze(1)
        score = torch.mul(u_score, i_score)
        return score

    def predict_interest_score(self, user, item, tag):
        u_emb = self.user_embeddings(user).unsqueeze(1)  # (B,E)->(B,1,E)
        i_emb = self.item_embeddings(item).unsqueeze(1)
        t_emb = self.interest_tag_embeddings.weight[tag].transpose(1, 2)  # (B,C,E)->(B,E,C)
        u_score = torch.bmm(u_emb, t_emb).squeeze(1)  # (B,1,C)->(B,C)
        i_score = torch.bmm(i_emb, t_emb).squeeze(1)
        score = torch.mul(u_score, i_score)
        return score
