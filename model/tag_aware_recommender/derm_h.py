# -*- coding: utf-8 -*-
# @Time   : 2023/02/16
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
DERM_H
################################################
We design a Deep Explainable Recommender Model, which models user/item representations using history selected tags.
"""

import torch
import torch.nn as nn

from ..layers import MLPLayers


class DERM_H(nn.Module):
    def __init__(self, config):
        super(DERM_H, self).__init__()

        self.tag_num = config['tag_num']
        self.candidate_num = config['candidate_num']
        self.tag_embeddings = nn.Embedding(config['tag_num'], config['embedding_size'])
        self.user_reason_list = config['user_reason_list']
        self.user_video_list = config['user_video_list']
        self.user_interest_list = config['user_interest_list']
        self.item_tag_list = config['item_tag_list']

        # Network
        # mlp_hidden_size [64,32]: 3 layers
        self.transfer_layers = MLPLayers([config['embedding_size'] * 2] + config['mlp_hidden_size'],
                                         config['dropout_prob'])
        self.predict_rating_layer = nn.Linear(config['mlp_hidden_size'][-1], 1)
        self.predict_reason_layer = nn.Linear(config['embedding_size'] + config['mlp_hidden_size'][-1], 1)
        self.predict_video_layer = nn.Linear(config['embedding_size'] + config['mlp_hidden_size'][-1], 1)
        self.predict_interest_layer = nn.Linear(config['embedding_size'] + config['mlp_hidden_size'][-1], 1)

        self.bce_loss = nn.BCELoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()
        self.device = config['device']

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

    def get_user_embedding(self, user, tag_type):
        if tag_type == 0:
            user_emb = [torch.mean(self.tag_embeddings(torch.LongTensor(list(self.user_reason_list[u])).to(self.device)), dim=0) for u in user]
        elif tag_type == 1:
            user_emb = [torch.mean(self.tag_embeddings(torch.LongTensor(list(self.user_video_list[u])).to(self.device)), dim=0) for u in user]
        else:
            user_emb = [torch.mean(self.tag_embeddings(torch.LongTensor(list(self.user_interest_list[u])).to(self.device)), dim=0) for u in user]
        return torch.stack(user_emb, dim=0)

    def get_item_embedding(self, item):
        item_emb = [torch.mean(self.tag_embeddings(torch.LongTensor(list(self.item_tag_list[i])).to(self.device)), dim=0) for i in item]
        return torch.stack(item_emb, dim=0)

    def get_mid_state(self, user, item, tag_type):
        u_emb = self.get_user_embedding(user, tag_type)
        i_emb = self.get_item_embedding(item)
        mid_state = self.transfer_layers(torch.cat((u_emb, i_emb), dim=1))
        return mid_state

    def predict_rating(self, user, item):
        mid_state0 = self.get_mid_state(user, item, 0)
        rating0 = self.predict_rating_layer(mid_state0)
        mid_state1 = self.get_mid_state(user, item, 1)
        rating1 = self.predict_rating_layer(mid_state1)
        mid_state2 = self.get_mid_state(user, item, 2)
        rating2 = self.predict_rating_layer(mid_state2)
        rating = (rating0 + rating1 + rating2) / 3
        return rating.squeeze(-1)

    def predict_reason_score(self, user, item, candi_tags):
        '''
        Calculate the score of (u,i) pair on all tags
        '''
        mid_state = self.get_mid_state(user, item, 0)  # (B, E)
        tag_emb_expand = self.tag_embeddings.weight[candi_tags]  # (B,T,E)
        input = torch.cat((mid_state.unsqueeze(1).repeat(1, self.candidate_num, 1), tag_emb_expand), dim=-1)
        reason_score = self.sigmoid(self.predict_reason_layer(input)).reshape(-1, self.candidate_num)  # (B,T,1)->(B,T)
        return reason_score

    def predict_video_score(self, user, item, candi_tags):
        mid_state = self.get_mid_state(user, item, 1)
        tag_emb_expand = self.tag_embeddings.weight[candi_tags]  # (B,T,E)
        input = torch.cat((mid_state.unsqueeze(1).repeat(1, self.candidate_num, 1), tag_emb_expand), dim=-1)
        video_score = self.sigmoid(self.predict_video_layer(input)).reshape(-1, self.candidate_num)  # (B,T,1)->(B,T)
        return video_score

    def predict_interest_score(self, user, item, candi_tags):
        mid_state = self.get_mid_state(user, item, 2)
        tag_emb_expand = self.tag_embeddings.weight[candi_tags]  # (B,T,E)
        input = torch.cat((mid_state.unsqueeze(1).repeat(1, self.candidate_num, 1), tag_emb_expand), dim=-1)
        interest_score = self.sigmoid(self.predict_interest_layer(input)).reshape(-1, self.candidate_num)  # (B,T,1)->(B,T)
        return interest_score

    def predict_specific_tag_score(self, user, item, tag, tag_type):
        mid_state = self.get_mid_state(user, item, tag_type)
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
        # print('user,item',user, item)
        # print('1  ', reason_score)
        # print('2  ', reason_label)
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
        l2_loss = self.tag_embeddings.weight.norm(2)
        return l2_loss
