import sys
sys.path.append('..')

from transformers import BertTokenizerFast, BertModel
import pandas as pd

plm_path = '/mnt/huggingface_models/bert-base-chinese-rm'
threshold = 2

def do_expand(thres: int = threshold):
    """
    注意调用模型的时候需要使用
    model.resize_token_embeddings(len(tokenizer))
    来更新embedding矩阵
    :param thres:
    :return:
    """
    tokenizer = BertTokenizerFast.from_pretrained(plm_path)

    unk_df = pd.read_csv('temp_data/unk_words_df.csv')
    words = unk_df[unk_df['count'] >= thres]['word'].to_list()

    num_added_toks = tokenizer.add_tokens(words)
    tokenizer.save_pretrained(plm_path)


if __name__ == '__main__':
    do_expand()

