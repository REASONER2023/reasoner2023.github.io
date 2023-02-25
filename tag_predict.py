# -*- coding: utf-8 -*-
# @Time   : 2023/02/07
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import os
import torch
import argparse

from config import Config
from utils import now_time, set_seed, get_model, get_trainer, get_dataloader, get_batchify

parser = argparse.ArgumentParser(description='Tag Prediction')

parser.add_argument('--model', '-m', type=str, default='DERM_MLP',
                    help='base model name')
parser.add_argument('--dataset', '-d', type=str, default='3_core',
                    help='dataset name')
parser.add_argument('--config', '-c', type=str, default='DERM_MLP.yaml',
                    help='config files')
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
# device = torch.device('cpu')

device = torch.device('cuda' if config['cuda'] else 'cpu')
if config['cuda']:
    torch.cuda.set_device(config['gpu_id'])

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

train_data = get_batchify(config['model_type'], config['model'], config['train_type'], 'train')(corpus.trainset, config,
                                                                                                tag_num, shuffle=True)
val_data = get_batchify(config['model_type'], config['model'], config['train_type'], 'valid')(corpus.validset, config,
                                                                                              tag_num)
test_data = get_batchify(config['model_type'], config['model'], config['train_type'], 'test')(corpus.testset, config,
                                                                                              tag_num)

# Bulid the user-item & user-tag & item-tag interaction matrix based on trainset
if config['model'] == 'EFM' or config['model'] == 'AMF':
    X_r, Y_r, X_v, Y_v, X_i, Y_i = corpus.build_inter_matrix(model_name=config['model'])
    config['X_r'] = X_r.to(device)
    config['Y_r'] = Y_r.to(device)
    config['X_v'] = X_v.to(device)
    config['Y_v'] = Y_v.to(device)
    config['X_i'] = X_i.to(device)
    config['Y_i'] = Y_i.to(device)
if config['model'] == 'DERM_H':
    config['user_reason_list'], config['user_video_list'], config['user_interest_list'], \
    config['item_tag_list'] = corpus.build_history_interaction()

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
# Build the model
###############################################################################

model = get_model(config['model'])(config).to(device)
trainer = get_trainer(config['model_type'], config['model'])(config, model, train_data, val_data)
###############################################################################
# Loop over epochs
###############################################################################

model_path, best_epoch = trainer.train_loop()

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)
print(now_time() + 'Load the best model' + model_path)

# Run on test data.
rmse, mse, \
reason_p, reason_r, reason_f1, reason_ndcg, \
video_p, video_r, video_f1, video_ndcg, \
interest_p, interest_r, interest_f1, interest_ndcg = trainer.evaluate(model, test_data)
print('=' * 89)
# Results
print('Best model in epoch {}'.format(best_epoch))
print('Best results: RMSE {:7.4f} | MAE {:7.4f}'.format(rmse, mse))
print('Best test: RMSE {:7.4f} | MAE {:7.4f}'.format(rmse, mse))
print('Best test: reason_tag   @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.5f} | ndcg {:7.5f}'
      .format(config['top_k'], reason_p, reason_r, reason_f1, reason_ndcg))
print('Best test: video_tag    @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.5f} | ndcg {:7.5f}'
      .format(config['top_k'], video_p, video_r, video_f1, video_ndcg))
print('Best test: interest_tag @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.5f} | ndcg {:7.5f}'
      .format(config['top_k'], interest_p, interest_r, interest_f1, interest_ndcg))
