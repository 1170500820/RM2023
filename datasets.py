import sys
sys.path.append('..')

import json

import torch
from torch.utils.data import Dataset


class RM_Dataset(Dataset):
    def __init__(self, data_type: str, tokenizer, overfit: bool = False):
        if data_type == 'dev' or data_type == 'val':
            data_type = 'valid'
        if data_type == 'train':
            fname = '../data/processed/proc_train.json'
        elif data_type == 'valid':
            fname = '../data/processed/proc_valid.json'
        else:
            raise Exception(f'[RM_Dataset]未知的数据类型：{data_type}！')

        self.tokenizer = tokenizer
        self.raw_data = json.load(open(fname, 'r', encoding='utf-8'))
        for e in self.raw_data:
            tokenized = tokenizer(e['text'])
            e.update({
                'input_ids': tokenized['input_ids'],
                'token_type_ids': tokenized['token_type_ids'],
                'attention_mask': tokenized['attention_mask'],
            })

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        return self.raw_data[index]




class RM_Auto_Dataset(Dataset):
    """
    能够自动执行数据与标签划分的数据集类
    """