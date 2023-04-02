''' 
_*_ coding: utf-8 _*_
Date: 2021/3/11
Author: 
Intent:
'''

import torch
import json
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class ProsodyDataset(Dataset):
    def __init__(self, config, file_path):
        super().__init__()
        self.config = config
        self.max_len = self.config['max_len']
        self.tokenizer = BertTokenizer.from_pretrained(self.config['pretrain_model_dir'])
        self.data = self._load_file(file_path)

    def _load_file(self, file_path):
        new_data = []
        with open(file_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        for instance in data:
            tokens = self.tokenizer.tokenize(instance['text'])
            labels = instance['prosody_label']
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
                labels = labels[:self.max_len]
            if len(tokens) == len(labels):
                new_data.append({'tokens': tokens, 'labels': labels})
        print(f'existing {len(data)} instances, loading {len(new_data)} instances...')
        return new_data

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens, labels = sample['tokens'], sample['labels']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return {"token_ids": token_ids,
                "label": labels}

    def __len__(self):
        return len(self.data)


def prosody_collate(data):
    batch_len = max([len(sample['token_ids']) for sample in data])
    inputs_ids, inputs_masks, tokens_type_ids, labels = [], [], [], []

    for i, sample in enumerate(data):
        token_ids, label = sample['token_ids'], sample['label']

        input_masks = [1] * len(token_ids) + [0] * (batch_len - len(token_ids))
        token_type_ids = [0] * batch_len
        input_ids = token_ids + [0] * (batch_len - len(token_ids))
        label += [0] * (batch_len - len(token_ids))

        inputs_ids.append(input_ids)
        inputs_masks.append(input_masks)
        tokens_type_ids.append(token_type_ids)
        labels.append(label)

    return {"inputs_ids": torch.LongTensor(inputs_ids),
            "inputs_masks": torch.LongTensor(inputs_masks),
            "tokens_type_ids": torch.LongTensor(tokens_type_ids),
            "labels": torch.LongTensor(labels)}


def test():
    from config import config
    dataset = ProsodyDataset(config, config['train_file'])
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=prosody_collate)
    for i, batch in enumerate(loader):
        print(batch)
        if i > 2:
            break



if __name__ == "__main__":
    test()
