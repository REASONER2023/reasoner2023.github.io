# -*- coding: utf-8 -*-
# @Time   : 2023/02/07
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import os
import torch
import torch.optim as optim
from utils import now_time, get_local_time, ids2tokens
from metrics.metrics import root_mean_square_error, mean_absolute_error, evaluate_precision_recall_f1, evaluate_ndcg, \
    bleu_score, rouge_score


class Trainer(object):

    def __init__(self, config, model, train_data, val_data):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

        self.model_name = config['model']
        self.dataset = config['dataset']
        self.epochs = config['epochs']

        self.device = config['device']
        self.batch_size = config['batch_size']
        self.learner = config['learner']
        self.learning_rate = config['lr']
        self.weight_decay = config['weight_decay']
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.95)  # gamma: lr_decay
        self.rating_weight = config['rating_weight']
        self.reason_weight = config['reason_weight']
        self.video_weight = config['video_weight']
        self.interest_weight = config['interest_weight']
        self.l2_weight = config['l2_weight']
        self.top_k = config['top_k']
        self.max_rating = config['max_rating']
        self.min_rating = config['min_rating']

        self.endure_times = config['endure_times']
        self.checkpoint = config['checkpoint']

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def train(self, data):  # train mse+bce+l2
        self.model.train()
        loss_sum = 0.
        Rating_loss = 0.
        Tag_loss = 0.
        L2_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, reason_tag, reason_label, video_tag, video_label, interest_tag, interest_label = data.next_batch()
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            reason_tag = reason_tag.to(self.device)
            video_tag = video_tag.to(self.device)
            interest_tag = interest_tag.to(self.device)
            reason_label = reason_label.to(self.device)
            video_label = video_label.to(self.device)
            interest_label = interest_label.to(self.device)

            self.optimizer.zero_grad()
            rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
            reason_loss = self.model.calculate_reason_loss(user, item, reason_tag, reason_label) * self.reason_weight
            video_loss = self.model.calculate_video_loss(user, item, video_tag, video_label) * self.video_weight
            interest_loss = self.model.calculate_interest_loss(user, item, interest_tag,
                                                               interest_label) * self.interest_weight
            l2_loss = self.model.calculate_l2_loss() * self.l2_weight
            loss = rating_loss + reason_loss + video_loss + interest_loss + l2_loss
            loss.backward()
            self.optimizer.step()
            Rating_loss += self.batch_size * rating_loss.item()
            Tag_loss += self.batch_size * (reason_loss.item() + video_loss.item() + interest_loss.item())
            L2_loss += self.batch_size * l2_loss.item()
            loss_sum += self.batch_size * loss.item()
            total_sample += self.batch_size

            if data.step == data.total_step:
                break
        return Rating_loss / total_sample, Tag_loss / total_sample, L2_loss / total_sample, loss_sum / total_sample

    def valid(self, data):  # valid
        self.model.eval()
        loss_sum = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, reason_tag, reason_label, video_tag, video_label, interest_tag, interest_label = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                reason_tag = reason_tag.to(self.device)
                video_tag = video_tag.to(self.device)
                interest_tag = interest_tag.to(self.device)
                reason_label = reason_label.to(self.device)
                video_label = video_label.to(self.device)
                interest_label = interest_label.to(self.device)

                rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
                reason_loss = self.model.calculate_reason_loss(user, item, reason_tag,
                                                               reason_label) * self.reason_weight
                video_loss = self.model.calculate_video_loss(user, item, video_tag, video_label) * self.video_weight
                interest_loss = self.model.calculate_interest_loss(user, item, interest_tag,
                                                                   interest_label) * self.interest_weight
                l2_loss = self.model.calculate_l2_loss() * self.l2_weight
                loss = rating_loss + reason_loss + video_loss + interest_loss + l2_loss

                loss_sum += self.batch_size * loss.item()
                total_sample += self.batch_size

                if data.step == data.total_step:
                    break
        return loss_sum / total_sample

    def evaluate(self, model, data):  # test
        model.eval()
        rating_predict = []
        reason_predict = []
        video_predict = []
        interest_predict = []
        with torch.no_grad():
            while True:
                user, item, candi_reason_tag, candi_video_tag, candi_interest_tag = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                reason_candidate_tag = candi_reason_tag.to(self.device)
                video_candidate_tag = candi_video_tag.to(self.device)
                interest_candidate_tag = candi_interest_tag.to(self.device)

                rating_p = model.predict_rating(user, item)  # (batch_size,)
                rating_predict.extend(rating_p.tolist())

                reason_p = model.predict_reason_score(user, item, reason_candidate_tag)  # (batch_size, candidate_num)
                _, reason_p_topk = torch.topk(reason_p, dim=-1, k=self.top_k, largest=True,
                                              sorted=True)  # values & index
                reason_predict.extend(reason_candidate_tag.gather(1, reason_p_topk).tolist())

                video_p = model.predict_video_score(user, item, video_candidate_tag)  # (batch_size,candidate_num)
                _, video_p_topk = torch.topk(video_p, dim=-1, k=self.top_k, largest=True, sorted=True)  # values & index
                video_predict.extend(video_candidate_tag.gather(1, video_p_topk).tolist())

                interest_p = model.predict_interest_score(user, item,
                                                          interest_candidate_tag)  # (batch_size,candidate_num)
                _, interest_p_topk = torch.topk(interest_p, dim=-1, k=self.top_k, largest=True,
                                                sorted=True)  # values & index
                interest_predict.extend(interest_candidate_tag.gather(1, interest_p_topk).tolist())

                if data.step == data.total_step:
                    break
        # rating
        rating_zip = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
        RMSE = root_mean_square_error(rating_zip, self.max_rating, self.min_rating)
        MAE = mean_absolute_error(rating_zip, self.max_rating, self.min_rating)
        # reason_tag
        reason_p, reason_r, reason_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_reason_tag,
                                                                     reason_predict)
        reason_ndcg = evaluate_ndcg(self.top_k, data.positive_reason_tag, reason_predict)
        # video_tag
        video_p, video_r, video_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_video_tag, video_predict)
        video_ndcg = evaluate_ndcg(self.top_k, data.positive_video_tag, video_predict)
        # interest_tag
        interest_p, interest_r, interest_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_interest_tag,
                                                                           interest_predict)
        interest_ndcg = evaluate_ndcg(self.top_k, data.positive_interest_tag, interest_predict)

        return RMSE, MAE, \
               reason_p, reason_r, reason_f1, reason_ndcg, \
               video_p, video_r, video_f1, video_ndcg, \
               interest_p, interest_r, interest_f1, interest_ndcg

    def train_loop(self):
        best_val_loss = float('inf')
        best_epoch = 0
        endure_count = 0
        model_path = ''
        for epoch in range(1, self.epochs + 1):
            print(now_time() + 'epoch {}'.format(epoch))
            train_r_loss, train_t_loss, train_l_loss, train_sum_loss = self.train(self.train_data)
            print(
                now_time() + 'rating loss {:4.4f} | tag loss {:4.4f} | l2 loss {:4.4f} |total loss {:4.4f} on train'.format(
                    train_r_loss, train_t_loss, train_l_loss, train_sum_loss))
            val_loss = self.valid(self.val_data)
            print(now_time() + 'total loss {:4.4f} on validation'.format(val_loss))

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_val_loss:
                saved_model_file = '{}-{}-{}.pt'.format(self.model_name, self.dataset, get_local_time())
                model_path = os.path.join(self.checkpoint, saved_model_file)
                with open(model_path, 'wb') as f:
                    torch.save(self.model, f)
                print(now_time() + 'Save the best model' + model_path)
                best_val_loss = val_loss
                best_epoch = epoch
            else:
                endure_count += 1
                print(now_time() + 'Endured {} time(s)'.format(endure_count))
                if endure_count == self.endure_times:
                    print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                    break
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                self.scheduler.step()
                print(now_time() + 'Learning rate set to {:2.8f}'.format(self.scheduler.get_last_lr()[0]))

        return model_path, best_epoch


class LRPPMTrainer(Trainer):
    def __init__(self, config, model, train_data, val_data):
        super(LRPPMTrainer, self).__init__(config, model, train_data, val_data)

    def train(self, data):
        self.model.train()
        loss_sum = 0.
        Rating_loss = 0.
        Tag_loss = 0.
        L2_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, reason_pos, reason_neg, video_pos, video_neg, interest_pos, interest_neg = data.next_batch()

            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            reason_pos = reason_pos.to(self.device)
            reason_neg = reason_neg.to(self.device)
            video_pos = video_pos.to(self.device)
            video_neg = video_neg.to(self.device)
            interest_pos = interest_pos.to(self.device)
            interest_neg = interest_neg.to(self.device)

            self.optimizer.zero_grad()
            rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
            reason_loss = self.model.calculate_tag_loss(user, item, reason_pos, reason_neg, 0) * self.reason_weight
            video_loss = self.model.calculate_tag_loss(user, item, video_pos, video_neg, 1) * self.video_weight
            interest_loss = self.model.calculate_tag_loss(user, item, interest_pos, interest_neg,
                                                          2) * self.interest_weight
            l2_loss = self.model.calculate_l2_loss() * self.l2_weight
            loss = rating_loss + reason_loss + video_loss + interest_loss + l2_loss

            loss.backward()
            self.optimizer.step()
            Rating_loss += self.batch_size * rating_loss.item()
            Tag_loss += self.batch_size * (reason_loss.item() + video_loss.item() + interest_loss.item())
            L2_loss += self.batch_size * l2_loss.item()
            loss_sum += self.batch_size * loss.item()
            total_sample += self.batch_size

            if data.step == data.total_step:
                break
        return Rating_loss / total_sample, Tag_loss / total_sample, L2_loss / total_sample, loss_sum / total_sample

    def valid(self, data):  # valid
        self.model.eval()
        loss_sum = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, reason_pos, reason_neg, video_pos, video_neg, interest_pos, interest_neg = data.next_batch()

                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                reason_pos = reason_pos.to(self.device)
                reason_neg = reason_neg.to(self.device)
                video_pos = video_pos.to(self.device)
                video_neg = video_neg.to(self.device)
                interest_pos = interest_pos.to(self.device)
                interest_neg = interest_neg.to(self.device)

                rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
                reason_loss = self.model.calculate_tag_loss(user, item, reason_pos, reason_neg, 0) * self.reason_weight
                video_loss = self.model.calculate_tag_loss(user, item, video_pos, video_neg, 1) * self.video_weight
                interest_loss = self.model.calculate_tag_loss(user, item, interest_pos, interest_neg,
                                                              2) * self.interest_weight
                l2_loss = self.model.calculate_l2_loss() * self.l2_weight
                loss = rating_loss + reason_loss + video_loss + interest_loss + l2_loss

                loss_sum += self.batch_size * loss.item()
                total_sample += self.batch_size

                if data.step == data.total_step:
                    break
        return loss_sum / total_sample

    def evaluate(self, model, data):  # test
        model.eval()
        rating_predict = []
        reason_predict = []
        video_predict = []
        interest_predict = []
        with torch.no_grad():
            while True:
                user, item, candi_reason_tag, candi_video_tag, candi_interest_tag = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                reason_candidate_tag = candi_reason_tag.to(self.device)
                video_candidate_tag = candi_video_tag.to(self.device)
                interest_candidate_tag = candi_interest_tag.to(self.device)

                rating_p = model.predict_rating(user, item)  # (batch_size,)
                rating_predict.extend(rating_p.tolist())

                reason_p = model.rank_tags(user, item, reason_candidate_tag, 0)  # (batch_size, tag_num)
                _, reason_p_topk = torch.topk(reason_p, dim=-1, k=self.top_k, largest=True,
                                              sorted=True)  # values & index
                reason_predict.extend(reason_candidate_tag.gather(1, reason_p_topk).tolist())

                video_p = model.rank_tags(user, item, video_candidate_tag, 1)  # (batch_size,tag_num)
                _, video_p_topk = torch.topk(video_p, dim=-1, k=self.top_k, largest=True, sorted=True)  # values & index
                video_predict.extend(video_candidate_tag.gather(1, video_p_topk).tolist())

                interest_p = model.rank_tags(user, item, interest_candidate_tag, 2)  # (batch_size,tag_num)
                _, interest_p_topk = torch.topk(interest_p, dim=-1, k=self.top_k, largest=True,
                                                sorted=True)  # values & index
                interest_predict.extend(interest_candidate_tag.gather(1, interest_p_topk).tolist())

                if data.step == data.total_step:
                    break
        # rating
        rating_zip = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
        RMSE = root_mean_square_error(rating_zip, self.max_rating, self.min_rating)
        MAE = mean_absolute_error(rating_zip, self.max_rating, self.min_rating)
        # reason_tag
        reason_p, reason_r, reason_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_reason_tag,
                                                                     reason_predict)
        reason_ndcg = evaluate_ndcg(self.top_k, data.positive_reason_tag, reason_predict)
        # video_tag
        video_p, video_r, video_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_video_tag, video_predict)
        video_ndcg = evaluate_ndcg(self.top_k, data.positive_video_tag, video_predict)
        # interest_tag
        interest_p, interest_r, interest_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_interest_tag,
                                                                           interest_predict)
        interest_ndcg = evaluate_ndcg(self.top_k, data.positive_interest_tag, interest_predict)

        return RMSE, MAE, \
               reason_p, reason_r, reason_f1, reason_ndcg, \
               video_p, video_r, video_f1, video_ndcg, \
               interest_p, interest_r, interest_f1, interest_ndcg


class EFMTrainer(Trainer):
    def __init__(self, config, model, train_data, val_data):
        super(EFMTrainer, self).__init__(config, model, train_data, val_data)
        self.non_neg_weight = config['non_neg_weight']
        self.X_r = config['X_r']
        self.Y_r = config['Y_r']
        self.X_v = config['X_v']
        self.Y_v = config['Y_v']
        self.X_i = config['X_i']
        self.Y_i = config['Y_i']

    def train(self, data):  # train mse+bce+l2
        self.model.train()
        loss_sum = 0.
        Rating_loss = 0.
        Tag_loss = 0.
        L2_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, reason_tag, reason_label, video_tag, video_label, interest_tag, interest_label = data.next_batch()
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            user_reason_label = self.X_r[user]
            item_reason_label = self.Y_r[item]
            user_video_label = self.X_v[user]
            item_video_label = self.Y_v[item]
            user_interest_label = self.X_i[user]
            item_interest_label = self.Y_i[item]

            self.optimizer.zero_grad()
            rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
            reason_loss = self.model.calculate_reason_loss(user, item, user_reason_label,
                                                           item_reason_label) * self.reason_weight
            video_loss = self.model.calculate_video_loss(user, item, user_video_label,
                                                         item_video_label) * self.video_weight
            interest_loss = self.model.calculate_interest_loss(user, item, user_interest_label,
                                                               item_interest_label) * self.interest_weight
            non_neg_loss = self.model.calculate_non_negative_reg() * self.non_neg_weight
            l2_loss = self.model.calculate_l2_loss() * self.l2_weight
            loss = rating_loss + reason_loss + video_loss + interest_loss + l2_loss + non_neg_loss

            loss.backward()
            self.optimizer.step()
            Rating_loss += self.batch_size * rating_loss.item()
            Tag_loss += self.batch_size * (reason_loss.item() + video_loss.item() + interest_loss.item())
            L2_loss += self.batch_size * l2_loss.item()
            loss_sum += self.batch_size * loss.item()
            total_sample += self.batch_size

            if data.step == data.total_step:
                break
        return Rating_loss / total_sample, Tag_loss / total_sample, L2_loss / total_sample, loss_sum / total_sample

    def valid(self, data):  # valid
        self.model.eval()
        loss_sum = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, reason_tag, reason_label, video_tag, video_label, interest_tag, interest_label = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                user_reason_label = self.X_r[user]
                item_reason_label = self.Y_r[item]
                user_video_label = self.X_v[user]
                item_video_label = self.Y_v[item]
                user_interest_label = self.X_i[user]
                item_interest_label = self.Y_i[item]

                rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
                reason_loss = self.model.calculate_reason_loss(user, item, user_reason_label,
                                                               item_reason_label) * self.reason_weight
                video_loss = self.model.calculate_video_loss(user, item, user_video_label,
                                                             item_video_label) * self.video_weight
                interest_loss = self.model.calculate_interest_loss(user, item, user_interest_label,
                                                                   item_interest_label) * self.interest_weight
                non_neg_loss = self.model.calculate_non_negative_reg() * self.non_neg_weight
                l2_loss = self.model.calculate_l2_loss() * self.l2_weight
                loss = rating_loss + reason_loss + video_loss + interest_loss + l2_loss + non_neg_loss

                loss_sum += self.batch_size * loss.item()
                total_sample += self.batch_size

                if data.step == data.total_step:
                    break
        return loss_sum / total_sample


class AMFTrainer(Trainer):
    def __init__(self, config, model, train_data, val_data):
        super(AMFTrainer, self).__init__(config, model, train_data, val_data)
        self.X_r = config['X_r']
        self.Y_r = config['Y_r']
        self.X_v = config['X_v']
        self.Y_v = config['Y_v']
        self.X_i = config['X_i']
        self.Y_i = config['Y_i']

    def train(self, data):  # train mse+bce+l2
        self.model.train()
        loss_sum = 0.
        Rating_loss = 0.
        Tag_loss = 0.
        L2_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, reason_tag, reason_label, video_tag, video_label, interest_tag, interest_label = data.next_batch()
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            user_reason_label = self.X_r[user]
            item_reason_label = self.Y_r[item]
            user_video_label = self.X_v[user]
            item_video_label = self.Y_v[item]
            user_interest_label = self.X_i[user]
            item_interest_label = self.Y_i[item]

            self.optimizer.zero_grad()
            rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
            reason_loss = self.model.calculate_reason_loss(user, item, user_reason_label,
                                                           item_reason_label) * self.reason_weight
            video_loss = self.model.calculate_video_loss(user, item, user_video_label,
                                                         item_video_label) * self.video_weight
            interest_loss = self.model.calculate_interest_loss(user, item, user_interest_label,
                                                               item_interest_label) * self.interest_weight
            l2_loss = self.model.calculate_l2_loss() * self.l2_weight
            loss = rating_loss + reason_loss + video_loss + interest_loss + l2_loss

            loss.backward()
            self.optimizer.step()
            Rating_loss += self.batch_size * rating_loss.item()
            Tag_loss += self.batch_size * (reason_loss.item() + video_loss.item() + interest_loss.item())
            L2_loss += self.batch_size * l2_loss.item()
            loss_sum += self.batch_size * loss.item()
            total_sample += self.batch_size

            if data.step == data.total_step:
                break
        return Rating_loss / total_sample, Tag_loss / total_sample, L2_loss / total_sample, loss_sum / total_sample

    def valid(self, data):  # valid
        self.model.eval()
        loss_sum = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, reason_tag, reason_label, video_tag, video_label, interest_tag, interest_label = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                user_reason_label = self.X_r[user]
                item_reason_label = self.Y_r[item]
                user_video_label = self.X_v[user]
                item_video_label = self.Y_v[item]
                user_interest_label = self.X_i[user]
                item_interest_label = self.Y_i[item]

                rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
                reason_loss = self.model.calculate_reason_loss(user, item, user_reason_label,
                                                               item_reason_label) * self.reason_weight
                video_loss = self.model.calculate_video_loss(user, item, user_video_label,
                                                             item_video_label) * self.video_weight
                interest_loss = self.model.calculate_interest_loss(user, item, user_interest_label,
                                                                   item_interest_label) * self.interest_weight
                l2_loss = self.model.calculate_l2_loss() * self.l2_weight
                loss = rating_loss + reason_loss + video_loss + interest_loss + l2_loss

                loss_sum += self.batch_size * loss.item()
                total_sample += self.batch_size

                if data.step == data.total_step:
                    break
        return loss_sum / total_sample


class MTERTrainer(Trainer):
    def __init__(self, config, model, train_data, val_data):
        super(MTERTrainer, self).__init__(config, model, train_data, val_data)
        self.non_neg_weight = config['non_neg_weight']

    def train(self, data):
        self.model.train()
        loss_sum = 0.
        Rating_loss = 0.
        Tag_loss = 0.
        L2_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, reason_pos, reason_neg, video_pos, video_neg, interest_pos, interest_neg = data.next_batch()

            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            reason_pos = reason_pos.to(self.device)
            reason_neg = reason_neg.to(self.device)
            video_pos = video_pos.to(self.device)
            video_neg = video_neg.to(self.device)
            interest_pos = interest_pos.to(self.device)
            interest_neg = interest_neg.to(self.device)

            self.optimizer.zero_grad()
            rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
            reason_loss = self.model.calculate_reason_loss(user, item, reason_pos, reason_neg) * self.reason_weight
            video_loss = self.model.calculate_video_loss(user, item, video_pos, video_neg) * self.video_weight
            interest_loss = self.model.calculate_interest_loss(user, item, interest_pos,
                                                               interest_neg) * self.interest_weight
            non_neg_loss = self.model.calculate_non_negative_reg() * self.non_neg_weight
            loss = rating_loss + reason_loss + video_loss + interest_loss + non_neg_loss

            loss.backward()
            self.optimizer.step()
            Rating_loss += self.batch_size * rating_loss.item()
            Tag_loss += self.batch_size * (reason_loss.item() + video_loss.item() + interest_loss.item())
            loss_sum += self.batch_size * loss.item()
            total_sample += self.batch_size

            if data.step == data.total_step:
                break
        return Rating_loss / total_sample, Tag_loss / total_sample, L2_loss / total_sample, loss_sum / total_sample

    def valid(self, data):  # valid
        self.model.eval()
        loss_sum = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, reason_pos, reason_neg, video_pos, video_neg, interest_pos, interest_neg = data.next_batch()

                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                reason_pos = reason_pos.to(self.device)
                reason_neg = reason_neg.to(self.device)
                video_pos = video_pos.to(self.device)
                video_neg = video_neg.to(self.device)
                interest_pos = interest_pos.to(self.device)
                interest_neg = interest_neg.to(self.device)

                rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
                reason_loss = self.model.calculate_reason_loss(user, item, reason_pos, reason_neg) * self.reason_weight
                video_loss = self.model.calculate_video_loss(user, item, video_pos, video_neg) * self.video_weight
                interest_loss = self.model.calculate_interest_loss(user, item, interest_pos,
                                                                   interest_neg) * self.interest_weight
                non_neg_loss = self.model.calculate_non_negative_reg() * self.non_neg_weight
                loss = rating_loss + reason_loss + video_loss + interest_loss + non_neg_loss

                loss_sum += self.batch_size * loss.item()
                total_sample += self.batch_size

                if data.step == data.total_step:
                    break
        return loss_sum / total_sample

    def evaluate(self, model, data):  # test
        model.eval()
        rating_predict = []
        reason_predict = []
        video_predict = []
        interest_predict = []
        with torch.no_grad():
            while True:
                user, item, candi_reason_tag, candi_video_tag, candi_interest_tag = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                reason_candidate_tag = candi_reason_tag.to(self.device)
                video_candidate_tag = candi_video_tag.to(self.device)
                interest_candidate_tag = candi_interest_tag.to(self.device)

                rating_p = model.predict_rating(user, item)  # (batch_size,)
                rating_predict.extend(rating_p.tolist())

                reason_p = model.rank_reason_tags(user, item, reason_candidate_tag)  # (batch_size, tag_num)
                _, reason_p_topk = torch.topk(reason_p, dim=-1, k=self.top_k, largest=True,
                                              sorted=True)  # values & index
                reason_predict.extend(reason_candidate_tag.gather(1, reason_p_topk).tolist())

                video_p = model.rank_video_tags(user, item, video_candidate_tag)  # (batch_size,tag_num)
                _, video_p_topk = torch.topk(video_p, dim=-1, k=self.top_k, largest=True, sorted=True)  # values & index
                video_predict.extend(video_candidate_tag.gather(1, video_p_topk).tolist())

                interest_p = model.rank_interest_tags(user, item, interest_candidate_tag)  # (batch_size,tag_num)
                _, interest_p_topk = torch.topk(interest_p, dim=-1, k=self.top_k, largest=True,
                                                sorted=True)  # values & index
                interest_predict.extend(interest_candidate_tag.gather(1, interest_p_topk).tolist())

                if data.step == data.total_step:
                    break
        # rating
        rating_zip = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
        RMSE = root_mean_square_error(rating_zip, self.max_rating, self.min_rating)
        MAE = mean_absolute_error(rating_zip, self.max_rating, self.min_rating)
        # reason_tag
        reason_p, reason_r, reason_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_reason_tag,
                                                                     reason_predict)
        reason_ndcg = evaluate_ndcg(self.top_k, data.positive_reason_tag, reason_predict)
        # video_tag
        video_p, video_r, video_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_video_tag, video_predict)
        video_ndcg = evaluate_ndcg(self.top_k, data.positive_video_tag, video_predict)
        # interest_tag
        interest_p, interest_r, interest_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_interest_tag,
                                                                           interest_predict)
        interest_ndcg = evaluate_ndcg(self.top_k, data.positive_interest_tag, interest_predict)

        return RMSE, MAE, \
               reason_p, reason_r, reason_f1, reason_ndcg, \
               video_p, video_r, video_f1, video_ndcg, \
               interest_p, interest_r, interest_f1, interest_ndcg


class Att2SeqTrainer(object):
    def __init__(self, config, model, train_data, val_data):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

        self.model_name = config['model']
        self.dataset = config['dataset']
        self.epochs = config['epochs']

        self.device = config['device']
        self.batch_size = config['batch_size']
        self.learner = config['learner']
        self.learning_rate = config['lr']
        self.weight_decay = config['weight_decay']
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.95)  # gamma: lr_decay
        self.max_rating = config['max_rating']
        self.min_rating = config['min_rating']
        self.endure_times = config['endure_times']
        self.checkpoint = config['checkpoint']
        self.ntokens = config['token_num']
        self.text_criterion = config['text_criterion']
        self.seq_max_len = config['seq_max_len']
        self.word2idx = config['word2idx']
        self.idx2word = config['idx2word']
        self.pad_idx = config['pad_idx']
        self.clip = config['clip']

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, eps=1e-3)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def train(self, data):
        self.model.train()
        text_loss = 0.
        total_sample = 0
        while True:
            user, item, _, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            seq = seq.to(self.device)  # (batch_size, seq_len + 2)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.optimizer.zero_grad()
            log_word_prob = self.model(user, item, seq[:, :-1])  # (batch_size,) vs. (batch_size, seq_len + 1, ntoken)
            loss = self.text_criterion(log_word_prob.view(-1, self.ntokens), seq[:, 1:].reshape((-1,)))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
        return text_loss / total_sample

    def valid(self, data):
        self.model.eval()
        text_loss = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, _, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
                batch_size = user.size(0)
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                seq = seq.to(self.device)  # (batch_size, seq_len + 2)
                log_word_prob = self.model(user, item,
                                           seq[:, :-1])  # (batch_size,) vs. (batch_size, seq_len + 1, ntoken)
                loss = self.text_criterion(log_word_prob.view(-1, self.ntokens), seq[:, 1:].reshape((-1,)))

                text_loss += batch_size * loss.item()
                total_sample += batch_size

                if data.step == data.total_step:
                    break
        return text_loss / total_sample

    def evaluate(self, model, data):
        model.eval()
        idss_predict = []
        RMSE = 0.
        MSE = 0.
        with torch.no_grad():
            while True:
                user, item, _, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                inputs = seq[:, :1].to(self.device)  # (batch_size, 1)
                hidden = None
                hidden_c = None
                ids = inputs
                for idx in range(self.seq_max_len):
                    # produce a word at each step
                    if idx == 0:
                        hidden = model.encoder(user, item)
                        hidden_c = torch.zeros_like(hidden)
                        log_word_prob, hidden, hidden_c = model.decoder(inputs, hidden,
                                                                        hidden_c)  # (batch_size, 1, ntoken)
                    else:
                        log_word_prob, hidden, hidden_c = model.decoder(inputs, hidden,
                                                                        hidden_c)  # (batch_size, 1, ntoken)
                    word_prob = log_word_prob.squeeze().exp()  # (batch_size, ntoken)
                    inputs = torch.argmax(word_prob, dim=1,
                                          keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                    ids = torch.cat([ids, inputs], 1)  # (batch_size, len++)
                ids = ids[:, 1:].tolist()  # remove bos
                idss_predict.extend(ids)

                if data.step == data.total_step:
                    break

        # text
        tokens_test = [ids2tokens(ids[1:], self.word2idx, self.idx2word) for ids in data.seq.tolist()]
        tokens_predict = [ids2tokens(ids, self.word2idx, self.idx2word) for ids in idss_predict]
        BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
        print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
        BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
        print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))

        text_test = [' '.join(tokens) for tokens in tokens_test]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        ROUGE = rouge_score(text_test, text_predict)  # a dictionary
        for (k, v) in ROUGE.items():
            print(now_time() + '{} {:7.4f}'.format(k, v))
        text_out = ''
        for (real, fake) in zip(text_test, text_predict):
            text_out += '{}\n{}\n\n'.format(real, fake)
        return text_out, RMSE, MSE, BLEU1, BLEU4, ROUGE

    def train_loop(self):
        # Loop over epochs.
        best_epoch = 0
        best_val_loss = float('inf')
        endure_count = 0
        model_path = ''
        for epoch in range(1, self.epochs + 1):
            print(now_time() + 'epoch {}'.format(epoch))
            train_loss = self.train(self.train_data)
            print(now_time() + 'text loss {:4.4f} on train'.format(train_loss))
            val_loss = self.valid(self.val_data)
            print(now_time() + 'text loss {:4.4f} on validation'.format(val_loss))
            if val_loss < best_val_loss:
                saved_model_file = '{}-{}-{}.pt'.format(self.model_name, self.dataset, get_local_time())
                model_path = os.path.join(self.checkpoint, saved_model_file)
                with open(model_path, 'wb') as f:
                    torch.save(self.model, f)
                print(now_time() + 'Save the best model' + model_path)
                best_val_loss = val_loss
                best_epoch = epoch
            else:
                endure_count += 1
                print(now_time() + 'Endured {} time(s)'.format(endure_count))
                if endure_count == self.endure_times:
                    print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                    break
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                self.scheduler.step()
                print(now_time() + 'Learning rate set to {:2.8f}'.format(self.scheduler.get_last_lr()[0]))

        return model_path, best_epoch


class NRTTrainer(Att2SeqTrainer):
    def __init__(self, config, model, train_data, val_data):
        super(NRTTrainer, self).__init__(config, model, train_data, val_data)
        self.rating_weight = config['rating_weight']
        self.review_weight = config['review_weight']
        self.l2_weight = config['l2_weight']
        self.rating_criterion = config['rating_criterion']

    def train(self, data):
        self.model.train()
        text_loss = 0.
        rating_loss = 0.
        sum_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            seq = seq.to(self.device)  # (batch_size, seq_len + 2)
            self.optimizer.zero_grad()
            rating_p, log_word_prob = self.model(user, item,
                                                 seq[:, :-1])  # (batch_size,) vs. (batch_size, seq_len + 1, ntoken)
            r_loss = self.rating_criterion(rating_p, rating)
            t_loss = self.text_criterion(log_word_prob.view(-1, self.ntokens), seq[:, 1:].reshape((-1,)))
            l2_loss = torch.cat([x.view(-1) for x in self.model.parameters()]).pow(2.).sum()
            loss = self.review_weight * t_loss + self.rating_weight * r_loss + self.l2_weight * l2_loss
            loss.backward()
            self.optimizer.step()

            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            sum_loss += batch_size * loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
        return rating_loss / total_sample, text_loss / total_sample, sum_loss / total_sample

    def valid(self, data):
        self.model.eval()
        text_loss = 0.
        rating_loss = 0.
        sum_loss = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
                batch_size = user.size(0)
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                seq = seq.to(self.device)  # (batch_size, seq_len + 2)
                rating_p, log_word_prob = self.model(user, item,
                                                     seq[:, :-1])  # (batch_size,) vs. (batch_size, seq_len + 1, ntoken)
                r_loss = self.rating_criterion(rating_p, rating)
                t_loss = self.text_criterion(log_word_prob.view(-1, self.ntokens), seq[:, 1:].reshape((-1,)))
                l2_loss = torch.cat([x.view(-1) for x in self.model.parameters()]).pow(2.).sum()
                loss = self.review_weight * t_loss + self.rating_weight * r_loss + self.l2_weight * l2_loss

                text_loss += batch_size * t_loss.item()
                rating_loss += batch_size * r_loss.item()
                sum_loss += batch_size * loss.item()
                total_sample += batch_size

                if data.step == data.total_step:
                    break
        return sum_loss / total_sample

    def evaluate(self, model, data):
        model.eval()
        idss_predict = []
        rating_predict = []
        with torch.no_grad():
            while True:
                user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                inputs = seq[:, :1].to(self.device)  # (batch_size, 1)
                hidden = None
                ids = inputs
                for idx in range(self.seq_max_len):
                    # produce a word at each step
                    if idx == 0:
                        rating_p, hidden = model.encoder(user, item)
                        rating_predict.extend(rating_p.tolist())
                        log_word_prob, hidden = model.decoder(inputs, hidden)  # (batch_size, 1, ntoken)
                    else:
                        log_word_prob, hidden = model.decoder(inputs, hidden)  # (batch_size, 1, ntoken)
                    word_prob = log_word_prob.squeeze().exp()  # (batch_size, ntoken)
                    inputs = torch.argmax(word_prob, dim=1,
                                          keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                    ids = torch.cat([ids, inputs], 1)  # (batch_size, len++)
                ids = ids[:, 1:].tolist()  # remove bos
                idss_predict.extend(ids)

                if data.step == data.total_step:
                    break

        # rating
        predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
        RMSE = root_mean_square_error(predicted_rating, self.max_rating, self.min_rating)
        print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
        MAE = mean_absolute_error(predicted_rating, self.max_rating, self.min_rating)
        print(now_time() + 'MAE {:7.4f}'.format(MAE))
        # text
        tokens_test = [ids2tokens(ids[1:], self.word2idx, self.idx2word) for ids in data.seq.tolist()]
        tokens_predict = [ids2tokens(ids, self.word2idx, self.idx2word) for ids in idss_predict]
        # bleu
        BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
        print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
        BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
        print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
        # rouge
        text_test = [' '.join(tokens) for tokens in tokens_test]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        ROUGE = rouge_score(text_test, text_predict)  # a dictionary
        for (k, v) in ROUGE.items():
            print(now_time() + '{} {:7.4f}'.format(k, v))
        text_out = ''
        for (real, fake) in zip(text_test, text_predict):
            text_out += '{}\n{}\n\n'.format(real, fake)

        return text_out, RMSE, MAE, BLEU1, BLEU4, ROUGE

    def train_loop(self):
        # Loop over epochs.
        best_epoch = 0
        best_val_loss = float('inf')
        endure_count = 0
        model_path = ''
        for epoch in range(1, self.epochs + 1):
            print(now_time() + 'epoch {}'.format(epoch))
            train_r_loss, train_t_loss, train_loss = self.train(self.train_data)
            print(now_time() + 'rating loss {:4.4f} | text loss {:4.4f} | total loss {:4.4f} on train'.format(
                train_t_loss, train_r_loss, train_loss))
            val_loss = self.valid(self.val_data)
            print(now_time() + 'total loss {:4.4f} on validation'.format(val_loss))
            if val_loss < best_val_loss:
                saved_model_file = '{}-{}-{}.pt'.format(self.model_name, self.dataset, get_local_time())
                model_path = os.path.join(self.checkpoint, saved_model_file)
                with open(model_path, 'wb') as f:
                    torch.save(self.model, f)
                print(now_time() + 'Save the best model' + model_path)
                best_val_loss = val_loss
                best_epoch = epoch
            else:
                endure_count += 1
                print(now_time() + 'Endured {} time(s)'.format(endure_count))
                if endure_count == self.endure_times:
                    print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                    break
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                self.scheduler.step()
                print(now_time() + 'Learning rate set to {:2.8f}'.format(self.scheduler.get_last_lr()[0]))

        return model_path, best_epoch


class PETERTrainer(Att2SeqTrainer):
    def __init__(self, config, model, train_data, val_data):
        super(PETERTrainer, self).__init__(config, model, train_data, val_data)
        self.rating_weight = config['rating_weight']
        self.review_weight = config['review_weight']
        self.context_weight = config['context_weight']
        self.rating_criterion = config['rating_criterion']
        self.src_len = config['src_len']
        self.tgt_len = config['tgt_len']

    def predict(self, log_context_dis, topk):
        word_prob = log_context_dis.exp()  # (batch_size, ntoken)
        if topk == 1:
            context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
        else:
            context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
        return context  # (batch_size, topk)

    def train(self, data):  # train
        # Turn on training mode which enables dropout.
        self.model.train()
        context_loss = 0.
        text_loss = 0.
        rating_loss = 0.
        total_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            seq = seq.t().to(self.device)  # (tgt_len + 1, batch_size)     .t()
            text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.optimizer.zero_grad()

            log_word_prob, log_context_dis, rating_p, _ = self.model(user, item,
                                                                     text)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            context_dis = log_context_dis.unsqueeze(0).repeat(
                (self.tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)

            c_loss = self.text_criterion(context_dis.view(-1, self.ntokens), seq[1:-1].reshape((-1,)))
            r_loss = self.rating_criterion(rating_p, rating)
            t_loss = self.text_criterion(log_word_prob.view(-1, self.ntokens), seq[1:].reshape((-1,)))

            loss = self.rating_weight * r_loss + self.context_weight * c_loss + self.review_weight * t_loss
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            context_loss += batch_size * c_loss.item()
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_loss += batch_size * loss.item()
            total_sample += batch_size
            if data.step == data.total_step:
                break

        return rating_loss / total_sample, context_loss / total_sample, text_loss / total_sample, total_loss / total_sample

    def valid(self, data):  # valid and test based on loss
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        context_loss = 0.
        text_loss = 0.
        rating_loss = 0.
        total_loss = 0.
        total_sample = 0
        rating_predict = []
        with torch.no_grad():
            while True:
                user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
                batch_size = user.size(0)
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                seq = seq.t().to(self.device)  # (tgt_len + 1, batch_size)
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
                log_word_prob, log_context_dis, rating_p, _ = self.model(user, item,
                                                                         text)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                rating_predict.extend(rating_p.tolist())
                context_dis = log_context_dis.unsqueeze(0).repeat(
                    (self.tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
                c_loss = self.text_criterion(context_dis.view(-1, self.ntokens), seq[1:-1].reshape((-1,)))
                r_loss = self.rating_criterion(rating_p, rating)
                t_loss = self.text_criterion(log_word_prob.view(-1, self.ntokens), seq[1:].reshape((-1,)))
                loss = self.rating_weight * r_loss + self.context_weight * c_loss + self.review_weight * t_loss

                context_loss += batch_size * c_loss.item()
                text_loss += batch_size * t_loss.item()
                rating_loss += batch_size * r_loss.item()
                total_loss += batch_size * loss.item()
                total_sample += batch_size

                if data.step == data.total_step:
                    break

        return total_loss / total_sample

    def evaluate(self, model, data):  # generate explanation & evaluate on metrics
        # Turn on evaluation mode which disables dropout.
        model.eval()
        idss_predict = []
        context_predict = []
        rating_predict = []

        with torch.no_grad():
            while True:
                user, item, rating, seq = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                bos = seq[:, 0].unsqueeze(0).to(self.device)  # (1, batch_size)
                text = bos  # (src_len - 1, batch_size)
                start_idx = text.size(0)
                for idx in range(self.seq_max_len):
                    # produce a word at each step
                    if idx == 0:  # predict word from <bos>
                        log_word_prob, log_context_dis, rating_p, _ = model(user, item, text,
                                                                            False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                        rating_predict.extend(rating_p.tolist())
                        context = self.predict(log_context_dis, topk=self.seq_max_len)  # (batch_size, seq_max_len)
                        context_predict.extend(context.tolist())
                    else:
                        log_word_prob, _, _, _ = model(user, item, text, False, False, False)  # (batch_size, ntoken)
                    word_prob = log_word_prob.exp()  # (batch_size, ntoken)
                    word_idx = torch.argmax(word_prob,
                                            dim=1)  # (batch_size,), pick the one with the largest probability
                    text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
                ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
                idss_predict.extend(ids)

                if data.step == data.total_step:
                    break

        # rating
        predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
        RMSE = root_mean_square_error(predicted_rating, self.max_rating, self.min_rating)
        print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
        MAE = mean_absolute_error(predicted_rating, self.max_rating, self.min_rating)
        print(now_time() + 'MAE {:7.4f}'.format(MAE))
        # text
        tokens_test = [ids2tokens(ids[1:], self.word2idx, self.idx2word) for ids in data.seq.tolist()]
        tokens_predict = [ids2tokens(ids, self.word2idx, self.idx2word) for ids in idss_predict]
        BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
        print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
        BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
        print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))

        text_test = [' '.join(tokens) for tokens in tokens_test]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        tokens_context = [' '.join([self.idx2word[i] for i in ids]) for ids in context_predict]
        ROUGE = rouge_score(text_test, text_predict)  # a dictionary
        for (k, v) in ROUGE.items():
            print(now_time() + '{} {:7.4f}'.format(k, v))
        text_out = ''
        for (real, ctx, fake) in zip(text_test, tokens_context,
                                     text_predict):  # format: ground_truth|context|explanation
            text_out += '{}\n{}\n{}\n\n'.format(real, ctx, fake)
        return text_out, RMSE, MAE, BLEU1, BLEU4, ROUGE

    def train_loop(self):
        # Loop over epochs.
        best_epoch = 0
        best_val_loss = float('inf')
        endure_count = 0
        model_path = ''
        for epoch in range(1, self.epochs + 1):
            print(now_time() + 'epoch {}'.format(epoch))
            train_r_loss, train_c_loss, train_t_loss, train_loss = self.train(self.train_data)
            print(
                now_time() + 'rating loss {:4.4f} | context loss {:4.4f} | text loss {:4.4f} | total loss {:4.4f} on train'.format(
                    train_r_loss, train_c_loss, train_t_loss, train_loss))
            val_loss = self.valid(self.val_data)
            print(now_time() + 'total loss {:4.4f} on validation'.format(val_loss))
            if val_loss < best_val_loss:
                saved_model_file = '{}-{}-{}.pt'.format(self.model_name, self.dataset, get_local_time())
                model_path = os.path.join(self.checkpoint, saved_model_file)
                with open(model_path, 'wb') as f:
                    torch.save(self.model, f)
                print(now_time() + 'Save the best model' + model_path)
                best_val_loss = val_loss
                best_epoch = epoch
            else:
                endure_count += 1
                print(now_time() + 'Endured {} time(s)'.format(endure_count))
                if endure_count == self.endure_times:
                    print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                    break
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                self.scheduler.step()
                print(now_time() + 'Learning rate set to {:2.8f}'.format(self.scheduler.get_last_lr()[0]))

        return model_path, best_epoch
