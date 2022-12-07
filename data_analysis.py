import json


def find_labels():
    fname = 'data/processed/train.jsonl'
    d = list(json.loads(x) for x in open(fname, 'r', encoding='utf-8').read().strip().split('\n'))
    labels = set()
    for e in d:
        for e_label in e['label']:
            labels.add(e_label)
    print(len(d))
    print(len(labels))
    # print(labels)


if __name__ == '__main__':
    find_labels()
