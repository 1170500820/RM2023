import sys
sys.path.append('..')

import json
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizerFast


data_path = 'data/filtered/processed_1021'

class RM_Dataset(Dataset):
    def __init__(self, data_type: str, tokenizer, overfit: bool = False, q: float = 1):
        if data_type == 'dev' or data_type == 'val':
            data_type = 'valid'
        if data_type == 'train':
            # fname = 'data/processed/proc_train.json'
            fname = os.path.join(data_path, 'proc_train.json')
        elif data_type == 'valid':
            # fname = 'data/processed/proc_valid.json'
            fname = os.path.join(data_path, 'proc_valid.json')
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
        if q == 1:
            self.data = self.raw_data
        else:
            self.resample(q)


    def resample(self, q: float):
        """
        基于p_j = \frac{n^q_j}{\sum^{C}_{i=1}n^q_i}
        计算对每个标签的样本概率，然后进行re-sample
        :param q:
        :return:
        """
        raw_pd = pd.DataFrame(self.raw_data)
        raw_pd['label_copy'] = raw_pd['label'].copy()
        raw_pd = raw_pd.explode('label_copy')

        total = raw_pd.shape[0]  # 总样本数
        total_weight = (raw_pd['label_copy'].value_counts() ** q).sum()
        elem_weight = (raw_pd['label_copy'].value_counts() ** q)
        weight = elem_weight / total_weight  # 每个标签的采样概率
        select_count = (weight * total + 0.5).apply(int) # 每个标签的采样个数

        re_sampler = iter(raw_pd.groupby('label_copy'))
        result = []
        for label_id, df in re_sampler:
            df = df.drop('label_copy', axis=1)
            res = df.sample(select_count[label_id], replace=True)
            result.extend(res.apply(lambda x: x.to_dict(), axis=1).tolist())
        self.data = result


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class RM_Auto_Dataset(Dataset):
    """
    能够自动执行数据与标签划分的数据集类
    """

if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    d = RM_Dataset('train', tokenizer)
    print(len(d))
    d.resample(0.5)
    print(len(d))
