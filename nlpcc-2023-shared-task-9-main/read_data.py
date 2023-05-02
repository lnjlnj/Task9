import json
data_list = []

with open('./datasets/datasets_train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.rstrip())
        data_list.append(data)

test = []
new_data = []
num = []
class_1 = []
for n in data_list:
    m = {'query':n['query'], 'replys':[]}
    for reply in n['replys']:
        num.append(reply['dislike'] + reply['like'])
        prob = reply['like']/(reply['dislike'] + reply['like'])
        if reply['dislike'] + reply['like'] >0:
            if prob > 0 and prob <= 0.25:
                m['replys'].append({'reply':reply['reply'], 'label':0, 'prob_label':prob})

            elif prob > 0.25 and prob <0.5:
                m['replys'].append({'reply': reply['reply'], 'label': 0, 'prob_label':prob})

            # elif prob == 0.5:
            #     m['replys'].append({'reply': reply['reply'], 'label': 2, 'prob_label':prob})
            elif prob > 0.5 and prob <= 0.75:
                if len(class_1) > 9000:
                    continue
                m['replys'].append({'reply': reply['reply'], 'label': 1, 'prob_label':prob})
                class_1.append(reply)
            elif prob > 0.75:
                if len(class_1) > 9000:
                    continue
                m['replys'].append({'reply': reply['reply'], 'label': 1, 'prob_label':prob})
                class_1.append(reply)

    new_data.append(m)

num_10 = []
for n in num:
    if n == 5:
        num_10.append(n)


# with open('exp_dataset/train_2class_split.json', 'w') as f:
#     json.dump(new_data, f)

print()