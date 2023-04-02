''' 
_*_ coding: utf-8 _*_
Date: 2021/3/10
Author: 
Intent:
'''

import re
import json
import random


def get_instances_from_raw_data(raw_file, instance_file):
    with open(raw_file, 'r', encoding='utf8') as f, open(instance_file, 'w', encoding='utf8') as fs:
        for i, line in enumerate(f):
            if i % 2 == 0:
                data = line.strip().split('\t')
                basename, text = data[0], ''.join(data[1:])
                text_s = re.split(r'#\d', text)
                new_text = ''.join(text_s)
                p_labels = []
                for t in text_s:
                    c = 0
                    for s in t:
                        if re.search(r'\w', s):
                            if c == 0:
                                p_labels.append(0)
                            else:
                                p_labels.append(1)
                            c += 1
                        else:
                            p_labels.append(2)
                assert len(new_text) == len(p_labels)
                instance = {'id': basename, 'text': new_text, 'prosody_label': p_labels}
                fs.write(json.dumps(instance, ensure_ascii=False))
                fs.write('\n')


def split_dataset(data_file, train_file, dev_file, ratio=0.95):
    all_data = []
    with open(data_file, 'r', encoding='utf8') as f:
        for line in f:
            all_data.append(json.loads(line.strip()))

    random.shuffle(all_data)

    boundary = int(ratio*len(all_data))
    train_data, dev_data = all_data[:boundary], all_data[boundary:]
    print(f'train: {len(train_data)}, dev: {len(dev_data)}')

    with open(train_file, 'w', encoding='utf8') as ft, open(dev_file, 'w', encoding='utf8') as fd:
        json.dump(train_data, ft, ensure_ascii=False, indent=2)
        json.dump(dev_data, fd, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raw_file = './data/raw_biaobei/000001-010000.txt'
    instance_file = './data/text_prosody.json'
    # get_instances_from_raw_data(raw_file, instance_file)

    data_file = './data/text_prosody.json'
    train_file = './data/text_prosody_train.json'
    dev_file = './data/text_prosody_dev.json'
    # split_dataset(data_file, train_file, dev_file, ratio=0.95)

    pass