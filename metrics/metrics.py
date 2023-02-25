# -*- coding: utf-8 -*-
# @Time   : 2023/02/14
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import math
import torch
from bert_score import score
from .rouge import rouge
from .bleu import compute_bleu


###############################################################################
# Recommendation Metrics
###############################################################################

def mean_absolute_error(predicted, max_r, min_r, mae=True):  # MSE ↓
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += (sub ** 2)

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):  # RMSE ↓
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


def evaluate_precision_recall_f1(top_k, user2items_test, user2items_top):  # (B, pos_num)  (B, k)

    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    for i in range(len(user2items_test)):
        rank_list = user2items_top[i]
        test_list = user2items_test[i]
        hits = len(set(test_list) & set(rank_list))
        pre = hits / top_k
        rec = hits / len(test_list)
        precision_sum += pre
        recall_sum += rec
        if (pre + rec) > 0:
            f1_sum += 2 * pre * rec / (pre + rec)

    precision = precision_sum / len(user2items_test)
    recall = recall_sum / len(user2items_test)
    f1 = f1_sum / len(user2items_test)

    return precision, recall, f1


def evaluate_ndcg(top_k, user2items_test, user2items_top):
    dcgs = [1 / math.log(i + 2) for i in range(top_k)]

    ndcg = 0
    for i in range(len(user2items_test)):
        rank_list = user2items_top[i]
        test_list = user2items_test[i]
        dcg_u = 0
        for idx, item in enumerate(rank_list):
            if item in test_list:
                dcg_u += dcgs[idx]
        ndcg += dcg_u

    return ndcg / (sum(dcgs) * len(user2items_test))


###############################################################################
# Text Metrics
###############################################################################

def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def bert_score(references, generated, device, bert_bs):
    P, R, F1 = score(generated, references, lang="en", batch_size=bert_bs, rescale_with_baseline=True, device=device)  # len(pred)
    return torch.mean(F1) * 100, torch.mean(R) * 100, torch.mean(P) * 100