from TrainForClassify import ProbTrainer, CptForClassify, bert_processor,ProbDataset, tokenizer
import json
from torch.utils.data import DataLoader, Dataset
import torch

def change_data(data:list):
    new_data = []
    for n in data:
        for m in n['replys']:
            new_data.append({'query': f"用户：{n['query']}{tokenizer.cls_token}机器人：{m['reply']}", 'reply':m['reply'], 'label':m['label']})

    return new_data

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with open('/home/leiningjie/PycharmProjects/NLPCC-2023/Task9/nlpcc-2023-shared-task-9-main/my_datasets/train_classify.json', 'r') as f:
        train_data = json.load(f)
    with open('/home/leiningjie/PycharmProjects/NLPCC-2023/Task9/nlpcc-2023-shared-task-9-main/my_datasets/dev_classify.json', 'r') as f:
        dev_data = json.load(f)

    train_data = change_data(train_data)
    dev_data = change_data(dev_data)

    model = CptForClassify('/home/leiningjie/PycharmProjects/model/cpt/cpt-base')

    trainer = ProbTrainer(use_gpu=True,  train_dataset=train_data, eval_dataset=dev_data, model=model)
    # trainer.train(batch_size=18, accumulation_steps=2, total_epoches=100,learning_rate=0.00007)
    checkpoint = '/home/leiningjie/PycharmProjects/NLPCC-2023/Task9/nlpcc-2023-shared-task-9-main/CPT/CPT-master/log_classify/model_base/finetuned_model.bin'
    acc, records = trainer.evaluate_to_result(checkpoint_path=checkpoint)
    record_save_path = './record'
    # if os.path.exists(record_save_path) is False:
    #     os.makedirs(record_save_path)
    #
    # with open(f'{record_save_path}/record.json', 'w') as f:
    #     json.dump(records, f)

    print(acc)
