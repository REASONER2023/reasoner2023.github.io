# -*- coding: utf-8 -*-
# @Time   : 2023/02/13
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
Code Reference:
    https://github.com/RUCAIBox/RecBole
"""

import torch
import torch.nn as nn


class MLPLayers(nn.Module):
    r""" MLPLayers
    Examples:
        >>> m = MLPLayers([64, 32, 16], 0.2)
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(self, layers, dropout=0., bn=False):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)
