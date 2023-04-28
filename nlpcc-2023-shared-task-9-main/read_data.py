import json

data_list = []

with open('./datasets/datasets_dev.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.rstrip())
        data_list.append(data)

test = []
new_data = []
for n in data_list:
    m = {'query':n['query'], 'replys':[]}
    for reply in n['replys']:
        prob = reply['like']/(reply['dislike'] + reply['like'])
        if prob > 0.5:
            m['replys'].append({'reply':reply['reply'], 'label':1})
        elif prob < 0.5:
            m['replys'].append({'reply': reply['reply'], 'label': 0})
        elif prob == 0.5:
            m['replys'].append({'reply': reply['reply'], 'label': 2})

    new_data.append(m)

with open('./my_datasets/dev_classify.json', 'w') as f:
    json.dump(new_data, f)

print()