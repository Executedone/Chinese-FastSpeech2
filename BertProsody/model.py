''' 
_*_ coding: utf-8 _*_
Date: 2021/3/12
Author: 
Intent:
'''

from transformers import BertModel
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from datasets import ProsodyDataset, prosody_collate
import logging


logger = logging.getLogger(__name__)


class ProsodyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config['pretrain_model_dir'])
        self.hidden_size = self.bert.config.hidden_size
        self.proj = nn.Linear(self.hidden_size, 256)
        self.linear = nn.Linear(256, 3)

    def forward(self, inputs_ids, inputs_masks, tokens_type_ids):
        out_seq = self.bert(input_ids=inputs_ids,
                            attention_mask=inputs_masks,
                            token_type_ids=tokens_type_ids)[0]
        out_seq = self.proj(out_seq)
        out_seq = out_seq.reshape(inputs_ids.size(0)*inputs_ids.size(1), 256)
        logits = self.linear(out_seq)
        return logits


if __name__ == "__main__":
    from config import config
    model = ProsodyModel(config)
    dataset = ProsodyDataset(config, config['train_file'])
    loader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=2, collate_fn=prosody_collate)
    for i, batch in enumerate(loader):
        logits = model(batch['inputs_ids'], batch['inputs_masks'], batch['tokens_type_ids'])
        print(logits)
        if i > 5:
            break
