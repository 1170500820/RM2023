import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader

from datasets import RM_Dataset

from utils import io_tools, tools, batch_tool
from settings import RM_labels, RM_labels_idx


def RM_collate_fn(lst, padding=256):
    data_dict = tools.transpose_list_of_dict(lst)
    bsz = len(lst)

    # basic input
    input_ids = batch_tool.batchify_with_padding(data_dict['input_ids'], padding=padding).to(torch.long)
    token_type_ids = batch_tool.batchify_with_padding(data_dict['token_type_ids'], padding=padding).to(torch.long)
    attention_mask = batch_tool.batchify_with_padding(data_dict['attention_mask'], padding=padding).to(torch.long)
    max_length = input_ids.shape[1]

    labels = []
    for e in data_dict['label']:
        cur_label = torch.zeros(len(RM_labels))
        for e_label in e:
            cur_label[RM_labels_idx[e_label]] = 1
        labels.append(cur_label)
    label = torch.stack(labels)

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }, {
        'label': label
    }
