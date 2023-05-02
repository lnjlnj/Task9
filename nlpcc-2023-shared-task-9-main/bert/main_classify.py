from TrainForClassify import ProbTrainer, BertForClassify, bert_processor,ProbDataset, tokenizer
import json
from torch.utils.data import DataLoader, Dataset
import torch
from collections import Counter

def change_data(data:list):
    new_data = []
    class_num = []
    for n in data:
        for m in n['replys']:
            new_data.append({'query': f"{n['query']}{tokenizer.cls_token}{m['reply']}", 'reply':m['reply'], 'label':m['label']})
            class_num.append(m['label'])

    data_ration = Counter(class_num)
    ratio = []
    weights = []
    for category in range(len(data_ration)):
        percentage = data_ration[category] / len(class_num)
        ratio.append(1/percentage)


    return new_data, ratio

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with open('../my_datasets/train_5classify_prob.json', 'r') as f:
        train_data = json.load(f)
    with open('../my_datasets/dev_5classify_prob.json', 'r') as f:
        dev_data = json.load(f)

    train_data, data_ration = change_data(train_data)
    dev_data, _ = change_data(dev_data)

    model = BertForClassify('/home/leiningjie/PycharmProjects/model/bert/chinese-bert',
                            dataset_weights=torch.FloatTensor(data_ration).to('cuda'), num_class=5)
    trainer = ProbTrainer(use_gpu=True,  train_dataset=train_data, eval_dataset=dev_data, model=model)
    trainer.train(batch_size=16, accumulation_steps=2, total_epoches=100,learning_rate=0.00007)
    acc = trainer.evaluate()
    print(acc)
