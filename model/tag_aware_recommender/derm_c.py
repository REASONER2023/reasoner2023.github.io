# -*- coding: utf-8 -*-
# @Time   : 2023/02/14
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
DERM_C
################################################
We design a Deep Explainable Recommender Model, which combines deep neural network with matrix factorization.
"""

import torch
import torch.nn as nn

from ..layers import MLPLayers


class DERM_C(nn.Module):
    def __init__(self, config):
        super(DERM_C, self).__init__()

        self.tag_num = config['tag_num']
        self.candidate_num = config['candidate_num']
        self.user_embeddings_mf = nn.Embedding(config['user_num'], config['embedding_size_mf'])
        self.item_embeddings_mf = nn.Embedding(config['item_num'], config['embedding_size_mf'])
        self.user_embeddings_mlp = nn.Embedding(config['user_num'], config['embedding_size_mlp'])
        self.item_embeddings_mlp = nn.Embedding(config['item_num'], config['embedding_size_mlp'])
        self.tag_embeddings = nn.Embedding(config['tag_num'], config['embedding_size_mf'])

        # Network
        # mlp_hidden_size [64,32]: 3 layers
        self.transfer_layers = MLPLayers([config['embedding_size_mlp']*2] + config['mlp_hidden_size'], config['dropout_prob'])
        self.predict_rating_layer = nn.Linear(config['embedding_size_mf'] + config['mlp_hidden_size'][-1], 1)
        self.predict_reason_layer = nn.Linear(config['embedding_size_mf'] * 2 + config['mlp_hidden_size'][-1], 1)
        self.predict_video_layer = nn.Linear(config['embedding_size_mf'] * 2 + config['mlp_hidden_size'][-1], 1)
        self.predict_interest_layer = nn.Linear(config['embedding_size_mf'] * 2 + config['mlp_hidden_size'][-1], 1)

        self.bce_loss = nn.BCELoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=0.01)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self):
        return

    def get_mid_state(self, user, item):
        u_emb_mf = self.user_embeddings_mf(user)
        i_emb_mf = self.item_embeddings_mf(item)
        u_emb_mlp = self.user_embeddings_mlp(user)
        i_emb_mlp = self.item_embeddings_mlp(item)
        mf_output = torch.mul(u_emb_mf, i_emb_mf)  # [batch_size, embedding_size_mf]
        mlp_output = self.transfer_layers(torch.cat((u_emb_mlp, i_emb_mlp), -1))  # [batch_size, layers[-1]]
        return torch.cat((mf_output, mlp_output), -1)

    def predict_rating(self, user, item):
        mid_state = self.get_mid_state(user, item)
        rating = self.predict_rating_layer(mid_state)
        return rating.squeeze(-1)

    def predict_reason_score(self, user, item, candi_tags):
        '''
        Calculate the score of (u,i) pair on all tags
        '''
        mid_state = self.get_mid_state(user, item)  # (B, E)
        tag_emb_expand = self.tag_embeddings.weight[candi_tags]  # (B,T,E)
        input = torch.cat((mid_state.unsqueeze(1).repeat(1, self.candidate_num, 1), tag_emb_expand), dim=-1)
        reason_score = self.sigmoid(self.predict_reason_layer(input)).reshape(-1, self.candidate_num)  # (B,T,1)->(B,T)
        return reason_score

    def predict_video_score(self, user, item, candi_tags):
        mid_state = self.get_mid_state(user, item)
        tag_emb_expand = self.tag_embeddings.weight[candi_tags]  # (B,T,E)
        input = torch.cat((mid_state.unsqueeze(1).repeat(1, self.candidate_num, 1), tag_emb_expand), dim=-1)
        video_score = self.sigmoid(self.predict_video_layer(input)).reshape(-1, self.candidate_num)  # (B,T,1)->(B,T)
        return video_score

    def predict_interest_score(self, user, item, candi_tags):
        mid_state = self.get_mid_state(user, item)
        tag_emb_expand = self.tag_embeddings.weight[candi_tags]  # (B,T,E)
        input = torch.cat((mid_state.unsqueeze(1).repeat(1, self.candidate_num, 1), tag_emb_expand), dim=-1)
        interest_score = self.sigmoid(self.predict_interest_layer(input)).reshape(-1, self.candidate_num)  # (B,T,1)->(B,T)
        return interest_score

    def predict_specific_tag_score(self, user, item, tag, tag_type):
        mid_state = self.get_mid_state(user, item)
        tag_emb = self.tag_embeddings(tag)
        input = torch.cat((mid_state, tag_emb), dim=-1)
        if tag_type == 0:
            score = self.sigmoid(self.predict_reason_layer(input))
        elif tag_type == 1:
            score = self.sigmoid(self.predict_video_layer(input))
        else:
            score = self.sigmoid(self.predict_interest_layer(input))
        return score.squeeze(1)

    def calculate_rating_loss(self, user, item, rating_label):
        # MSEloss
        predicted_rating = self.predict_rating(user, item)
        rating_loss = self.mse_loss(predicted_rating, rating_label)
        return rating_loss

    def calculate_reason_loss(self, user, item, reason_tag, reason_label):
        # BCEloss
        reason_score = self.predict_specific_tag_score(user, item, reason_tag, tag_type=0)
        reason_loss = self.bce_loss(reason_score, reason_label)
        return reason_loss

    def calculate_video_loss(self, user, item, video_tag, video_label):
        # BCEloss
        video_score = self.predict_specific_tag_score(user, item, video_tag, tag_type=1)
        video_loss = self.bce_loss(video_score, video_label)
        return video_loss

    def calculate_interest_loss(self, user, item, interest_tag, interest_label):
        # BCEloss
        interest_score = self.predict_specific_tag_score(user, item, interest_tag, tag_type=2)
        interest_loss = self.bce_loss(interest_score, interest_label)
        return interest_loss

    def calculate_l2_loss(self):
        l2_loss = self.user_embeddings_mf.weight.norm(2) + self.item_embeddings_mf.weight.norm(2) + \
                  self.user_embeddings_mlp.weight.norm(2) + self.item_embeddings_mlp.weight.norm(2) + \
                  self.tag_embeddings.weight.norm(2)
        return l2_loss

