"""
先把数据转换成方便读取的模式
"""
"""
将比赛提供的csv数据集转换为jsonl格式
"""
import csv
import json
import os


train_data_path = 'data/weibo_topic_recognition_01/train.csv'
output_dir = 'data/processed'
output_fname = 'train.jsonl'


def convert_to_jsonl():
    """
    目前
    :return:
    """
    d = open(train_data_path, 'r', encoding='utf-8').read().strip().split('\n')
    rows = []
    for e in d[1:]:
        rows.append(e.split('\t'))
    result = []
    for e in rows:
        result.append({
            'id': e[0],
            'text': e[1],
            'label': e[2].split('，')
            # 'label_a': 0 if e[2] == 'sexist' else 1,
            # 'label_b': e[3],
            # 'label_c': e[4]
        })
    # 0 = neg = sexist
    # 1 = pos = not sexist

    p = os.path.join(output_dir, output_fname)
    f = open(p, 'w', encoding='utf-8')
    for e in result:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
    f.close()


def train_val_split(train_rate = 0.8):
    """
    做task a的时候可以随便划分
    但是task b & c需要考虑某些标签在train和val中的分布。
    :param train_rate:
    :return:
    """
    # 简单切分
    fname = os.path.join(output_dir, output_fname)
    d = list(json.loads(x) for x in open(fname, 'r', encoding='utf-8').read().strip().split('\n'))
    train_size = int(len(d) * train_rate)
    train_d, val_d = d[:train_size], d[train_size:]
    ftrain = os.path.join(output_dir, 'split_train.jsonl')
    fvalid = os.path.join(output_dir, 'split_valid.jsonl')

    f = open(ftrain, 'w', encoding='utf-8')
    for e in train_d:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
    f.close()
    f = open(fvalid, 'w', encoding='utf-8')
    for e in val_d:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
    f.close()

def convert_to_json():
    ftrain = 'data/filtered/processed/train.txt'
    fvalid = 'data/filtered/processed/valid.txt'

    dtrain = list(open(ftrain, 'r', encoding='utf-8').read().strip().split('\n'))
    train = list({'text': x[0], 'label': x[1].split(',')} for x in list(v.split('\t') for v in dtrain))

    dvalid = list(open(fvalid, 'r', encoding='utf-8').read().strip().split('\n'))
    valid = list({'text': x[0], 'label': x[1].split(',')} for x in list(v.split('\t') for v in dvalid))

    json.dump(train, open('data/filtered/processed/proc_train.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(valid, open('data/filtered/processed/proc_valid.json', 'w', encoding='utf-8'), ensure_ascii=False)


def generate_label_vocab():
    ftrain = 'data/filtered/processed/proc_train.json'
    fvalid = 'data/filtered/processed/proc_valid.json'

    dtrain = json.load(open(ftrain, 'r', encoding='utf-8'))
    dvalid = json.load(open(fvalid, 'r', encoding='utf-8'))

    label_set = set()
    for ev in [dtrain, dvalid]:
        for e in ev:
            for l in e:
                label_set.add(l)
    label_list = sorted(list(label_set))
    label_idx = {x: i for i, x in enumerate(label_list)}

    json.dump(label_list, open('data/filtered/processed/label_list.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(label_idx, open('data/filtered/processed/label_idx.json', 'w', encoding='utf-8'), ensure_ascii=False)

if __name__ == '__main__':
    # convert_to_jsonl()
    # train_val_split()
    # convert_to_json()
    generate_label_vocab()