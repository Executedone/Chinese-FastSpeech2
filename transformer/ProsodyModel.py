''' 
_*_ coding: utf-8 _*_
Date: 2022/2/4
Author: 
Intent:
'''

from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn as nn

# 加入prosody微调后的char enbedding
class CharEmbedding(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(model_dir)
        self.bert = BertModel(self.bert_config)
        self.hidden_size = self.bert_config.hidden_size
        self.proj = nn.Linear(self.hidden_size, 256)
        self.linear = nn.Linear(256, 3)

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

    def forward(self, inputs_ids, inputs_masks, tokens_type_ids):
        out_seq = self.bert(input_ids=inputs_ids,
                            attention_mask=inputs_masks,
                            token_type_ids=tokens_type_ids)[0]
        out_seq = self.proj(out_seq)

        return out_seq


if __name__ == "__main__":
    pass