# -*- coding: utf-8 -*-
# @Time   : 2023/02/14
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import os
import torch
import torch.nn as nn
import argparse

from config import Config
from utils import now_time, set_seed, get_model, get_trainer, get_dataloader, get_batchify

parser = argparse.ArgumentParser(description='Review Generation')

parser.add_argument('--model', '-m', type=str, default='NRT',
                    help='base model name')
parser.add_argument('--dataset', '-d', type=str, default='3_core',
                    help='dataset name')
parser.add_argument('--config', '-c', type=str, default='NRT.yaml',
                    help='config files')
args, _ = parser.parse_known_args()

config_file_list = args.config.strip().split(' ') if args.config else None
config = Config(config_file_list=config_file_list).final_config_dict
print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for param in config:
    print('{:40} {}'.format(param, config[param]))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

data_path = 'dataset/' + config['dataset'] + '/interaction.csv'
train_data_path = 'dataset/' + config['dataset'] + '/train.csv'
valid_data_path = 'dataset/' + config['dataset'] + '/valid.csv'
test_data_path = 'dataset/' + config['dataset'] + '/test.csv'
if data_path is None:
    parser.error('--data_path should be provided for loading data')
if not os.path.exists(config['checkpoint']):
    os.makedirs(config['checkpoint'])

model_path = ''
generated_file = args.dataset + config['generated_file_path']
prediction_path = os.path.join(config['checkpoint'], generated_file)

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
corpus = get_dataloader(config['model_type'])(data_path, train_data_path, valid_data_path, test_data_path,
                                              config['vocab_size'])
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
pad_idx = word2idx['<pad>']
user_num = corpus.user_num
item_num = corpus.item_num
token_num = len(corpus.word_dict)
trainset_size = corpus.train_size
validset_size = corpus.valid_size
testset_size = corpus.test_size
print(now_time() + '{}: user_num:{} | item_num:{} | token_num:{}'.format(config['dataset'], user_num, item_num,
                                                                         token_num))
print(now_time() + 'trainset:{} | validset:{} | testset:{}'.format(trainset_size, validset_size, testset_size))

train_data = get_batchify(config['model_type'], config['model'], config['train_type'], 'train')(corpus.trainset,
                                                                                                word2idx,
                                                                                                config['seq_max_len'],
                                                                                                config['batch_size'],
                                                                                                shuffle=True)
val_data = get_batchify(config['model_type'], config['model'], config['train_type'], 'valid')(corpus.validset, word2idx,
                                                                                              config['seq_max_len'],
                                                                                              config['batch_size'])
test_data = get_batchify(config['model_type'], config['model'], config['train_type'], 'test')(corpus.testset, word2idx,
                                                                                              config['seq_max_len'],
                                                                                              config['batch_size'])
text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
rating_criterion = nn.MSELoss()

###############################################################################
# Update Config
###############################################################################

config['user_num'] = user_num
config['item_num'] = item_num
config['token_num'] = token_num
config['max_rating'] = corpus.max_rating
config['min_rating'] = corpus.min_rating
config['device'] = device
config['word2idx'] = word2idx
config['idx2word'] = idx2word
config['text_criterion'] = text_criterion
config['rating_criterion'] = rating_criterion
config['src_len'] = 2
config['tgt_len'] = config['seq_max_len'] + 1
config['pad_idx'] = pad_idx

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
text_out, RMSE, MAE, BLEU1, BLEU4, ROUGE = trainer.evaluate(model, test_data)
print('=' * 89)
# Results
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))

print('Best model in epoch {}'.format(best_epoch))
if config['model'] != 'Att2Seq':
    print('Best test: RMSE {:7.4f} | MAE {:7.4f}'.format(RMSE, MAE))
print('Best test: BLEU1 {:7.4f} | BLEU4 {:7.4f}'.format(BLEU1, BLEU4))
for (k, v) in ROUGE.items():
    print('Best test: {} {:7.4f}'.format(k, v))
