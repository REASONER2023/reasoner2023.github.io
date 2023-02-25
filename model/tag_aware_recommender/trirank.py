# -*- coding: utf-8 -*-
# @Time   : 2023/02/07
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
TriRank
################################################
Reference:
    Xiangnan He et al. "TriRank: Review-aware Explainable Recommendation by Modeling Aspects." in CIKM 2015.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class TriRank(nn.Module):
    r"""TriRank is an algorithm that enriches the user–item binary relation to a user–item–aspect ternary relation.
    And it models the ternary relation as a heterogeneous tripartite graph.

    We adopt the ALS for optimization according to original paper.
    """

    def __init__(self, config):
        super(TriRank, self).__init__()

        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.tag_num = config['tag_num']

        # User-Item-Aspect Matrix
        self.userItem = config['userItem']
        self.userAspect = config['userAspect']
        self.itemAspect = config['itemAspect']

        self.userItem = self.symmetricNorm(self.userItem)
        self.userAspect = self.symmetricNorm(self.userAspect)
        self.itemAspect = self.symmetricNorm(self.itemAspect)

        self.itemUser = torch.transpose(self.userItem, 0, 1)
        self.aspectUser = torch.transpose(self.userAspect, 0, 1)
        self.aspectItem = torch.transpose(self.itemAspect, 0, 1)

        # Hyper-parameters
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.alpha0 = config['alpha0']
        self.beta0 = config['beta0']
        self.gamma0 = config['gamma0']
        self.maxIter = config['maxIter']
        self.device = config['device']

        self._init_weights()

    def _init_weights(self):
        return

    def symmetricNorm(self, w):
        r"""
        Symmetrically normalize a matrix.

        """
        row_weight = torch.sum(w, dim=1, keepdim=True)  # (R, 1)
        col_weight = torch.sum(w, dim=0, keepdim=True)  # (1, C)
        tot_weight = torch.mm(row_weight, col_weight)  # (R, C)
        S_w = w / (tot_weight + 1e-7)
        return S_w

    def init_Uscore(self, u):
        r"""Init Uscore for a user, use the current user.

        """
        Uscore = torch.zeros(self.user_num, 1)
        Uscore[u] = 1.0
        return Uscore.to(self.device)  # (user_num, 1)

    def init_Iscore(self, u):
        r"""Init Iscore for a user, use the current user.

        """
        Iscore = self.userItem[u]
        weight = Iscore.norm(1)
        Iscore = Iscore.view(-1, 1) / weight
        return Iscore.to(self.device)  # (item_num, 1)

    def init_Ascore(self, u):
        r"""Init Ascore for a user, use the current user.

        """
        Ascore = self.userAspect[u]
        weight = Ascore.norm(1)
        Ascore = Ascore.view(-1, 1) / weight
        return Ascore.to(self.device)  # (tag_num, 1)

    def forward(self):
        r"""Training

        """
        userItem_rank = torch.zeros(self.user_num, self.item_num)
        userAspect_rank = torch.zeros(self.user_num, self.tag_num)

        iter_data = tqdm(range(self.user_num), total=self.user_num, ncols=100, desc="Iteration process:")
        for u in iter_data:
            # for u in range(self.user_num):
            # print('Progress: {}/{}'.format(u, self.user_num))
            # Personalized score for users, items and aspects.
            Uscore0 = self.init_Uscore(u)
            Iscore0 = self.init_Iscore(u)
            Ascore0 = self.init_Ascore(u)
            # Initial rank scores for users, items and aspects.
            userVector = torch.rand(self.user_num, 1).to(self.device)
            itemVector = torch.rand(self.item_num, 1).to(self.device)
            aspectVector = torch.rand(self.tag_num, 1).to(self.device)
            # Iteration
            for cnt in range(self.maxIter):
                # Update for user rank scores.
                userVector = self.alpha / (self.alpha + self.gamma + self.alpha0) * torch.mm(self.userItem,
                                                                                             itemVector) + \
                             self.gamma / (self.alpha + self.gamma + self.alpha0) * torch.mm(self.userAspect,
                                                                                             aspectVector) + \
                             self.alpha0 / (self.alpha + self.gamma + self.alpha0) * Uscore0

                # Update for item rank scores.
                itemVector = self.alpha / (self.alpha + self.beta + self.beta0) * torch.mm(self.itemUser, userVector) + \
                             self.beta / (self.alpha + self.beta + self.beta0) * torch.mm(self.itemAspect,
                                                                                          aspectVector) + \
                             self.beta0 / (self.alpha + self.beta + self.beta0) * Iscore0
                # Update for aspect rank scores.
                aspectVector = self.alpha / (self.gamma + self.beta + self.gamma0) * torch.mm(self.aspectUser,
                                                                                              userVector) + \
                               self.gamma / (self.gamma + self.beta + self.gamma0) * torch.mm(self.aspectItem,
                                                                                              itemVector) + \
                               self.gamma0 / (self.gamma + self.beta + self.gamma0) * Ascore0
            userItem_rank[u] = itemVector.view(1, -1)
            userAspect_rank[u] = aspectVector.view(1, -1)

        return userItem_rank, userAspect_rank  # (nuser, nitem) (nuser, ntag)

    def predict_rating(self, userItem_rank, user, item):
        rating = userItem_rank[user, item]
        return rating

    def predict_aspect_score(self, userAspect_rank, user, tag):  # tag (B, C)
        aspect_score = userAspect_rank[user].to(self.device)  # (B, T)
        score = aspect_score.gather(1, tag)
        return score