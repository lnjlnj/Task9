import json

data_list = []

with open('./datasets/datasets_train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.rstrip())
        data_list.append(data)

print()