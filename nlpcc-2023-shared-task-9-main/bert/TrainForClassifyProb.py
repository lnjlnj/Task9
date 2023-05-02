import json
import faiss
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import logging
import os.path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import math
from transformers import AdamW, get_scheduler
from sklearn.metrics import accuracy_score, classification_report
from focal_loss.focal_loss import FocalLoss

tokenizer = BertTokenizer.from_pretrained('/home/leiningjie/PycharmProjects/model/bert/chinese-bert')



class ProbDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def kl_sim_score(gold, pred):
    score = 0
    for x, y in zip(gold, pred):
        x = max(x, 1e-6)
        x = min(x, 1 - 1e-6)
        y = max(y, 1e-6)
        y = min(y, 1 - 1e-6)
        sc1 = x * math.log(x) + (1 - x) * math.log(1 - x)
        sc2 = x * math.log(y) + (1 - x) * math.log(1 - y)
        score += 1 / (1 + sc1 - sc2)
    return score / len(gold)


def prepare_optimizer(model, lr, weight_decay, eps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        weight_decay,
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0,
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    return optimizer

def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler

def bert_processor(input:dict):
    result = {}
    # result['video_id'] = input['video_id']
    result['query'] = tokenizer(input['query'], padding=True, truncation=True, return_tensors='pt')
    result['reply'] = tokenizer(input['reply'],padding=True, truncation=True, return_tensors='pt')
    result['label'] = input['label']
    result['prob_label'] = input['prob_label']

    return result


class BertEncoder(nn.Module):
    def __init__(self, model_name):
        super(BertEncoder, self).__init__()
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, **encoded_input):
        outputs = self.model(**encoded_input)
        return outputs['pooler_output']


class BertForClassify(nn.Module):
    def __init__(self, bert_model_name, dataset_weights, num_class):
        super(BertForClassify, self).__init__()
        self.query_encoder = BertEncoder(bert_model_name)
        self.ClassifyNet = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_class),
            nn.Softmax(dim=-1)
        )

        self.loss_fct = FocalLoss(gamma=2, weights=dataset_weights)

        self.ProbNet = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.loss_fct_1 = nn.MSELoss()

    def forward(self, encode_input):
        label = encode_input['label'].long()
        prob_label = encode_input['prob_label'].float()
        query_vector = self.query_encoder(**encode_input['query'])
        mse_logits = self.ProbNet(query_vector)
        mse_loss = self.loss_fct_1(mse_logits.squeeze(dim=1), prob_label)
        logits = self.ClassifyNet(mse_logits)
        classify_logits = logits.squeeze(dim=1)
        cross_loss = self.loss_fct(logits.squeeze(dim=1), label)



        return classify_logits, mse_logits, cross_loss, mse_loss



class ProbTrainer:
    def __init__(self, model, use_gpu, *args, **kwargs):
        self.model = model
        self.use_gpu = use_gpu
        self.train_dataset = kwargs['train_dataset']
        self.eval_dataset = kwargs['eval_dataset']
        self.tokenizer = tokenizer

        if torch.cuda.is_available():
            if self.use_gpu:
                self.device = 'cuda'
                logging.info('训练将在GPU上进行...')

            else:
                self.device = 'cpu'
                logging.info('训练将在CPU上进行...')
        elif torch.cuda.is_available() is False:
            self.device = 'cpu'
            logging.info('找不到可用GPU,训练将在CPU上进行...')

    def train(self,
           model_savepath='./log_classify/model_classify_prob',
           eval_epoch=None,
           total_epoches=10,
           batch_size=16,
           accumulation_steps=1,
           learning_rate=1e-4,
           warmup_ratio=0.1,
           weight_decay=0.1,
           eps=1e-06,
           loss_log_freq=40):
        self.model.to(self.device)

        train_dataset = ProbDataset(self.train_dataset)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True)
        optimizer = prepare_optimizer(self.model, learning_rate,
                                      weight_decay, eps)
        steps_per_epoch = len(train_loader) // accumulation_steps
        scheduler = prepare_scheduler(optimizer, total_epoches,
                                      steps_per_epoch, warmup_ratio)

        eval_record = []
        losses = []

        for epoch in range(total_epoches):
            self.model.train()
            index = 0
            predicts = []
            labels = []
            prob_predicts = {}
            prob_labels = {}
            scores = []
            print(f'epoch{epoch}')
            for batch in tqdm(train_loader):

                input = bert_processor(batch)
                input['query'].to(self.device)
                input['reply'].to(self.device)
                input['label'] = input['label'].to(self.device)
                input['prob_label'] = input['prob_label'].to(self.device)

                classify_logits, mse_logits, cross_loss, mse_loss = self.model(input)
                pre = classify_logits.argmax(dim=-1).cpu().numpy()
                [predicts.append(pre[n]) for n in range(len(pre))]
                label = input['label'].to(self.device).cpu().numpy()
                [labels.append(label[n]) for n in range(len(label))]

                query = batch['query'][0]
                pre = mse_logits.squeeze(dim=-1).cpu().detach().numpy()
                label = input['prob_label'].to(self.device).cpu().numpy()
                if query not in prob_predicts.keys():
                    prob_predicts[query] = []
                if query not in prob_labels:
                    prob_labels[query] = []
                [prob_predicts[query].append(pre[n]) for n in range(len(pre))]
                [prob_labels[query].append(label[n]) for n in range(len(label))]

                if epoch > 3:
                    loss = mse_loss + 0.1*cross_loss
                else:
                    loss = cross_loss
                losses.append(loss)
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

                loss.backward()

                if (index + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                index += 1

            acc = classification_report(labels, predicts)
            result = accuracy_score(labels, predicts)
            print(f'--------train---------\n{acc}')
            for k in prob_predicts.keys():
                assert len(prob_predicts[k]) == len(prob_labels[k])
                scores.append(kl_sim_score(prob_predicts[k], prob_labels[k]))
            prob_result = sum(scores) / len(scores) * 100
            print(f'train prob_score: {prob_result}')

            classify_result, prob_result = self.evaluate()
            eval_record.append(classify_result)
            average_loss = sum(losses)/len(losses)
            print(f'epoch{epoch} acc: {classify_result}\naverage loss:{average_loss.item()}')


            losses = []

            if classify_result >= max(eval_record):
                state_dict = self.model.state_dict()
                if os.path.exists(model_savepath) is False:
                    os.makedirs(model_savepath)
                model_path = os.path.join(model_savepath,
                                          'finetuned_model.bin')

                torch.save(state_dict, model_path)
                print(
                    'epoch %d obtain max acc: %.4f, saving model_base to %s' %
                    (epoch, classify_result, model_path))


    def evaluate(self, per_gpu_batch_size=10, checkpoint_path=None):
        """
        Evaluate testsets
        """
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            self.model.load_state_dict(state_dict)

        eval_dataset = ProbDataset(self.eval_dataset)
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=1,
            shuffle=False)
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            classify_predicts = []
            classify_labels = []
            prob_predicts = {}
            prob_labels = {}
            scores = []
            for batch in tqdm(eval_loader):
                input = bert_processor(batch)
                input['query'].to(self.device)
                input['reply'].to(self.device)
                input['label'] = input['label'].to(self.device)
                input['prob_label'] = input['prob_label'].to(self.device)

                classify_logits, mse_logits, cross_loss, mse_loss = self.model(input)

                # classify
                classify_label = classify_logits.argmax(dim=-1)
                classify_predicts.append(classify_label.cpu().numpy().item())
                classify_labels.append(batch['label'].cpu().numpy().item())

                # prob
                query = batch['query'][0]
                if query not in prob_predicts.keys():
                    prob_predicts[query] = []
                if query not in prob_labels:
                    prob_labels[query] = []
                prob_predicts[query].append(mse_logits.cpu().numpy().item())
                prob_labels[query].append(input['prob_label'].cpu().numpy().item())

            # classify
            acc = classification_report(classify_labels, classify_predicts)
            classify_result = accuracy_score(classify_labels, classify_predicts)
            print(f'---------classify result-----------\n{acc}')

            # prob
            for k in prob_predicts.keys():
                assert len(prob_predicts[k]) == len(prob_labels[k])
                scores.append(kl_sim_score(prob_predicts[k], prob_labels[k]))
            prob_result = sum(scores) / len(scores) * 100
            print(f'prob_score: {prob_result}')

            return classify_result, prob_result







            # result_path = os.path.join(self.model_base.model_dir,
            #                            'evaluate_result.json')

