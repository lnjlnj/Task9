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
from transformers import AdamW, get_scheduler
import logging
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
import pyarrow.parquet as pq
import pandas as pd
import math
from transformers import AdamW, get_scheduler


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

    return result


class BertEncoder(nn.Module):
    def __init__(self, model_name):
        super(BertEncoder, self).__init__()
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, **encoded_input):
        outputs = self.model(**encoded_input)
        return outputs['pooler_output']


class BertForProb(nn.Module):
    def __init__(self, bert_model_name):
        super(BertForProb, self).__init__()
        self.query_encoder = BertEncoder(bert_model_name)
        self.keyword_encoder = BertEncoder(bert_model_name)
        self.ProbNet = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        self.loss_fct = nn.MSELoss()

    def forward(self, encode_input):
        label = encode_input['label'].float()
        query_vector = self.query_encoder(**encode_input['query'])
        reply_vector = self.keyword_encoder(**encode_input['reply'])
        vector = torch.cat((query_vector, reply_vector), dim=1)
        logits = self.ProbNet(vector).squeeze(dim=1)
        loss = self.loss_fct(logits, label)

        return logits, loss



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
           model_savepath='./log/model_base',
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
            print(f'epoch{epoch}')
            for batch in tqdm(train_loader):
                input = bert_processor(batch)
                input['query'].to(self.device)
                input['reply'].to(self.device)
                input['label'] = input['label'].to(self.device)
                logits, loss = self.model(input)

                losses.append(loss)
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

                loss.backward()

                if (index + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                index += 1

            result = self.evaluate()
            eval_record.append(result)
            average_loss = sum(losses)/len(losses)
            print(f'epoch{epoch} score: {result}\naverage loss:{average_loss.item()}')
            losses = []

            if result >= max(eval_record):
                state_dict = self.model.state_dict()
                if os.path.exists(model_savepath) is False:
                    os.makedirs(model_savepath)
                model_path = os.path.join(model_savepath,
                                          'finetuned_model.bin')

                torch.save(state_dict, model_path)
                print(
                    'epoch %d obtain max score: %.4f, saving model_base to %s' %
                    (epoch, result, model_path))


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
            predicts = {}
            labels = {}
            scores = []
            for batch in tqdm(eval_loader):
                query = batch['query'][0]
                if query not in predicts.keys():
                    predicts[query] = []
                if query not in labels:
                    labels[query] = []
                input = bert_processor(batch)
                input['query'].to(self.device)
                input['reply'].to(self.device)
                input['label'] = input['label'].to(self.device)
                logits, loss = self.model(input)
                predicts[query].append(logits.cpu().numpy().item())
                labels[query].append(input['label'].cpu().numpy().item())

            for k in predicts.keys():
                assert len(predicts[k]) == len(labels[k])
                scores.append(kl_sim_score(predicts[k], labels[k]))

            return sum(scores) / len(scores) * 100




            # result_path = os.path.join(self.model_base.model_dir,
            #                            'evaluate_result.json')

