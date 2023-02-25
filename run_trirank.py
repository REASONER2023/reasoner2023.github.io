# -*- coding: utf-8 -*-
# @Time   : 2023/02/07
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import os
import torch
import argparse

from config import Config
from utils import now_time, set_seed, get_model, get_dataloader, get_batchify
from metrics.metrics import root_mean_square_error, mean_absolute_error, evaluate_precision_recall_f1, evaluate_ndcg

parser = argparse.ArgumentParser(description='Tag Prediction for TriRank')

parser.add_argument('--model', '-m', type=str, default='TriRank',
                    help='base model name')
parser.add_argument('--dataset', '-d', type=str, default='3_core',
                    help='dataset name')
parser.add_argument('--config', '-c', type=str, default='TriRank.yaml',
                    help='config files')
parser.add_argument('--tag_type', '-t', type=int, default=0,
                    help='tag type:0 reason,1 video,2 interest')
args, _ = parser.parse_known_args()

config_file_list = args.config.strip().split(' ') if args.config else None
config = Config(config_file_list=config_file_list).final_config_dict
print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for param in config:
    print('{:40} {}'.format(param, config[param]))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

data_path = 'dataset/' + config['dataset'] + '/interaction.csv'
video_path = 'dataset/' + config['dataset'] + '/video.csv'
train_data_path = 'dataset/' + config['dataset'] + '/train.csv'
valid_data_path = 'dataset/' + config['dataset'] + '/valid.csv'
test_data_path = 'dataset/' + config['dataset'] + '/test.csv'
if data_path is None:
    parser.error('--data_path should be provided for loading data')
if not os.path.exists(config['checkpoint']):
    os.makedirs(config['checkpoint'])

# Set the random seed manually for reproducibility.
set_seed(config['seed'])
if torch.cuda.is_available():
    if not config['cuda']:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')

device = torch.device('cpu')
# device = torch.device('cuda' if config['cuda'] else 'cpu')
# if config['cuda']:
#     torch.cuda.set_device(config['gpu_id'])

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
corpus = get_dataloader(config['model_type'])(data_path, video_path, train_data_path, valid_data_path, test_data_path)
tag_num = corpus.tag_num
user_num = corpus.user_num
item_num = corpus.item_num
trainset_size = corpus.train_size
validset_size = corpus.valid_size
testset_size = corpus.test_size
print(now_time() + '{}: user_num:{} | item_num:{} | tag_num:{}'.format(config['dataset'], user_num, item_num, tag_num))
print(now_time() + 'trainset:{} | validset:{} | testset:{}'.format(trainset_size, validset_size, testset_size))

test_data = get_batchify(config['model_type'], config['model'], config['train_type'], 'test') \
    (corpus.testset, config, tag_num)
# Bulid the user-item & user-tag & item-tag interaction matrix based on trainset
type_dict = {0: 'reason_tag', 1: 'video_tag', 2: 'interest_tag'}
userItem, userAspect, itemAspect = corpus.bulid_TriRank_matrix(tag_type=type_dict[args.tag_type])
config['userItem'] = userItem.to(device)
config['userAspect'] = userAspect.to(device)
config['itemAspect'] = itemAspect.to(device)

###############################################################################
# Update Config
###############################################################################

config['user_num'] = user_num
config['item_num'] = item_num
config['tag_num'] = tag_num
config['max_rating'] = corpus.max_rating
config['min_rating'] = corpus.min_rating
config['device'] = device


###############################################################################
# Specific evaluate function
###############################################################################
def evaluate(data, userItem_rank, userAspect_rank, tag_type):
    # Turn on evaluation mode
    model.eval()
    rating_predict = []
    aspect_predict = []
    while True:
        user, item, candi_reason_tag, candi_video_tag, candi_interest_tag = data.next_batch()
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        if tag_type == 0:
            candidate_tag = candi_reason_tag.to(device)  # (B,C)
        elif tag_type == 1:
            candidate_tag = candi_video_tag.to(device)
        else:
            candidate_tag = candi_interest_tag.to(device)

        rating_p = model.predict_rating(userItem_rank, user, item)  # (batch_size,)
        rating_predict.extend(rating_p.tolist())

        aspect_p = model.predict_aspect_score(userAspect_rank, user, candidate_tag)  # (batch_size,)
        _, aspect_topk = torch.topk(aspect_p, dim=-1, k=config['top_k'], largest=True, sorted=True)  # values & index
        # (B, K)
        aspect_predict.extend(candidate_tag.gather(1, aspect_topk).tolist())

        if data.step == data.total_step:
            break
    # rating
    rating_zip = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
    RMSE = root_mean_square_error(rating_zip, config['max_rating'], config['min_rating'])
    MAE = mean_absolute_error(rating_zip, config['max_rating'], config['min_rating'])
    # tag
    if args.tag_type == 0:
        test_tag_pos = data.positive_reason_tag
    elif args.tag_type == 1:
        test_tag_pos = data.positive_video_tag
    else:
        test_tag_pos = data.positive_interest_tag
    precision, recall, f1 = evaluate_precision_recall_f1(config['top_k'], test_tag_pos, aspect_predict)
    ndcg = evaluate_ndcg(config['top_k'], test_tag_pos, aspect_predict)

    return RMSE, MAE, precision, recall, f1, ndcg


###############################################################################
# Build the model
###############################################################################
model = get_model(config['model'])(config).to(device)
userItem_rank, userAspect_rank = model()
rmse, mse, p, r, f1, ndcg, = evaluate(test_data, userItem_rank, userAspect_rank, args.tag_type)
print('=' * 89)

# Results
print('Best test: RMSE {:7.4f} | MAE {:7.4f}'.format(rmse, mse))
print('Best test: {}   @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.4f} | ndcg {:7.4f}'
      .format(type_dict[args.tag_type], config['top_k'], p, r, f1, ndcg))
