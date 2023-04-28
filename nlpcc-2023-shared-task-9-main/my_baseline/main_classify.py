from TrainForClassify import ProbTrainer, BertForClassify, bert_processor,ProbDataset
import json
from torch.utils.data import DataLoader, Dataset

def change_data(data:list):
    new_data = []
    for n in data:
        for m in n['replys']:
            new_data.append({'query':n['query'], 'reply':m['reply'], 'label':m['label']})

    return new_data

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with open('../my_datasets/train_classify.json', 'r') as f:
        train_data = json.load(f)
    with open('../my_datasets/dev_classify.json', 'r') as f:
        dev_data = json.load(f)

    train_data = change_data(train_data)
    dev_data = change_data(dev_data)

    model = BertForClassify('/home/leiningjie/PycharmProjects/model/bert/chinese-bert')
    trainer = ProbTrainer(use_gpu=True,  train_dataset=train_data, eval_dataset=dev_data, model=model)
    trainer.train(batch_size=32, accumulation_steps=2, total_epoches=100,learning_rate=0.0003)
    # acc = trainer.evaluate()
    # print(acc)
