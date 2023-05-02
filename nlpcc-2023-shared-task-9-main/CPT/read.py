import json

with open('./record/record.json', 'r') as f:
    data = json.load(f)

with open('/home/leiningjie/PycharmProjects/NLPCC-2023/Task9/nlpcc-2023-shared-task-9-main/my_datasets/dev.json', 'r') as f:
    dev = json.load(f)

corpus = {}
for n in dev:
    corpus[n['query']] = n['replys']



correct = []
fasle = []
for n in data:
    if n['predict'] == n['label']:
        correct.append(n)
    else:
        fasle.append(n)

for n in fasle:
    query = n['query'].split('用户：')[1]
    print(f'query:{query}\nresponse:{n["response"]}\npredict:{n["predict"]}  label:{n["label"]}\nlabel:{corpus[query]}\n')

print()