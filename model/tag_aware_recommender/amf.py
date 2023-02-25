# -*- coding: utf-8 -*-
# @Time   : 2023/02/09
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
AMF
################################################
Reference:
    Yunfeng Hou et al. "Explainable recommendation with fusion of aspect information." in WWW 2018.
"""

import torch
import torch.nn as nn


class AMF(nn.Module):
    r"""AMF is able to improve the accuracy of rating prediction by collaboratively decomposing the rating matrix with
    the auxiliary information extracted from aspects.

    """

    def __init__(self, config):
        super(AMF, self).__init__()

        self.candidate_num =config['candidate_num']
        self.user_embeddings = nn.Embedding(config['user_num'], config['embedding_size'])
        self.item_embeddings = nn.Embedding(config['item_num'], config['embedding_size'])
        self.reason_user_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])
        self.reason_item_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])
        self.video_user_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])
        self.video_item_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])
        self.interest_user_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])
        self.interest_item_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])

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
        t_u_emb = self.reason_user_embeddings.weight
        t_i_emb = self.reason_item_embeddings.weight
        ut_score = self.predict_tag_score(u_emb, t_u_emb)
        it_score = self.predict_tag_score(i_emb, t_i_emb)
        reason_loss = self.mse_loss(ut_score, user_reason_label) + self.mse_loss(it_score, item_reason_label)
        return reason_loss

    def calculate_video_loss(self, user, item, user_video_label, item_video_label):
        u_emb = self.user_embeddings(user)
        i_emb = self.item_embeddings(item)
        t_u_emb = self.video_user_embeddings.weight
        t_i_emb = self.video_item_embeddings.weight
        ut_score = self.predict_tag_score(u_emb, t_u_emb)
        it_score = self.predict_tag_score(i_emb, t_i_emb)
        video_loss = self.mse_loss(ut_score, user_video_label) + self.mse_loss(it_score, item_video_label)
        return video_loss

    def calculate_interest_loss(self, user, item, user_interest_label, item_interest_label):
        u_emb = self.user_embeddings(user)
        i_emb = self.item_embeddings(item)
        t_u_emb = self.interest_user_embeddings.weight
        t_i_emb = self.interest_item_embeddings.weight
        ut_score = self.predict_tag_score(u_emb, t_u_emb)
        it_score = self.predict_tag_score(i_emb, t_i_emb)
        interest_loss = self.mse_loss(ut_score, user_interest_label) + self.mse_loss(it_score, item_interest_label)
        return interest_loss

    def calculate_l2_loss(self):
        l2_loss = self.user_embeddings.weight.norm(2) + \
                  self.item_embeddings.weight.norm(2) + \
                  self.reason_user_embeddings.weight.norm(2) + \
                  self.reason_item_embeddings.weight.norm(2) + \
                  self.video_user_embeddings.weight.norm(2) + \
                  self.video_item_embeddings.weight.norm(2) + \
                  self.interest_user_embeddings.weight.norm(2) + \
                  self.interest_item_embeddings.weight.norm(2)
        return l2_loss

    def predict_reason_score(self, user, item, tag):
        u_emb = self.user_embeddings(user).unsqueeze(1)  # (B,E)->(B,1,E)
        i_emb = self.item_embeddings(item).unsqueeze(1)
        t_u_emb = self.reason_user_embeddings.weight[tag].transpose(1, 2)  # (B,C,E)->(B,E,C)
        t_i_emb = self.reason_item_embeddings.weight[tag].transpose(1, 2)
        u_score = torch.bmm(u_emb, t_u_emb).squeeze(1)  # (B,1,C)->(B,C)
        i_score = torch.bmm(i_emb, t_i_emb).squeeze(1)
        score = torch.mul(u_score, i_score)
        return score

    def predict_video_score(self, user, item, tag):
        u_emb = self.user_embeddings(user).unsqueeze(1)  # (B,E)->(B,1,E)
        i_emb = self.item_embeddings(item).unsqueeze(1)
        t_u_emb = self.video_user_embeddings.weight[tag].transpose(1, 2)  # (B,C,E)->(B,E,C)
        t_i_emb = self.video_item_embeddings.weight[tag].transpose(1, 2)
        u_score = torch.bmm(u_emb, t_u_emb).squeeze(1)  # (B,1,C)->(B,C)
        i_score = torch.bmm(i_emb, t_i_emb).squeeze(1)
        score = torch.mul(u_score, i_score)
        return score

    def predict_interest_score(self, user, item, tag):
        u_emb = self.user_embeddings(user).unsqueeze(1)  # (B,E)->(B,1,E)
        i_emb = self.item_embeddings(item).unsqueeze(1)
        t_u_emb = self.interest_user_embeddings.weight[tag].transpose(1, 2)  # (B,C,E)->(B,E,C)
        t_i_emb = self.interest_item_embeddings.weight[tag].transpose(1, 2)
        u_score = torch.bmm(u_emb, t_u_emb).squeeze(1)  # (B,1,C)->(B,C)
        i_score = torch.bmm(i_emb, t_i_emb).squeeze(1)
        score = torch.mul(u_score, i_score)
        return score


