# -*- coding: utf-8 -*-
# @Time   : 2023/02/13
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
NRT
################################################
Paper Reference:
    Piji Li et al. "Neural Rating Regression with Abstractive Tips Generation for Recommendation." in SIGIR 2017.
Code Reference:
    https://github.com/lileipisces/NRT
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as func


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class NRTEncoder(nn.Module):
    def __init__(self, nuser, nitem, emsize, hidden_size, num_layers=4, max_r=5, min_r=1):
        super(NRTEncoder, self).__init__()
        self.max_r = int(max_r)
        self.min_r = int(min_r)
        self.num_rating = self.max_r - self.min_r + 1

        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.first_layer = nn.Linear(emsize * 2, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()

        self.encoder = nn.Linear(emsize * 2 + self.num_rating, hidden_size)
        self.tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()

    def forward(self, user, item):  # (batch_size,)
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)
        ui_concat = torch.cat([u_src, i_src], 1)  # (batch_size, emsize * 2)
        hidden = self.sigmoid(self.first_layer(ui_concat))  # (batch_size, hidden_size)
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  # (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(hidden))  # (batch_size,)

        rating_int = torch.clamp(rating, min=self.min_r, max=self.max_r).type(torch.int64)  # (batch_size,)
        rating_one_hot = func.one_hot(rating_int - self.min_r, num_classes=self.num_rating)  # (batch_size, num_rating)

        encoder_input = torch.cat([u_src, i_src, rating_one_hot], 1)  # (batch_size, emsize * 2 + num_rating)
        encoder_state = self.tanh(self.encoder(encoder_input)).unsqueeze(0)  # (1, batch_size, hidden_size)

        return rating, encoder_state  # (batch_size,) vs. (1, batch_size, hidden_size)


class GRUDecoder(nn.Module):
    def __init__(self, ntoken, emsize, hidden_size):
        super(GRUDecoder, self).__init__()
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.gru = nn.GRU(emsize, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, seq, hidden):  # seq: (batch_size, seq_len), hidden: (nlayers, batch_size, hidden_size)
        seq_emb = self.word_embeddings(seq)  # (batch_size, seq_len, emsize)\
        output, hidden = self.gru(seq_emb, hidden)  # (batch_size, seq_len, hidden_size) vs. (nlayers, batch_size, hidden_size)
        decoded = self.linear(output)  # (batch_size, seq_len, ntoken)
        return func.log_softmax(decoded, dim=-1), hidden


class NRT(nn.Module):
    def __init__(self, config):
        super(NRT, self).__init__()
        self.encoder = NRTEncoder(config['user_num'], config['item_num'], config['embedding_size'], config['hidden_size'],
                                  config['nlayers'], config['max_rating'], config['min_rating'])
        self.decoder = GRUDecoder(config['token_num'], config['embedding_size'], config['hidden_size'])

    def forward(self, user, item, seq):  # (batch_size,) vs. (batch_size, seq_len)
        rating, hidden = self.encoder(user, item)
        log_word_prob, _ = self.decoder(seq, hidden)
        return rating, log_word_prob  # (batch_size,) vs. (batch_size, seq_len, ntoken)
