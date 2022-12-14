"""
基于简单的BERT实现一个微博话题分类模型
"""
import sys
sys.path.append('..')

from typing import List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup, BertPreTrainedModel

from datasets import RM_Dataset
from collate_functions import RM_collate_fn
from settings import RM_labels_idx, RM_labels


class MutilLabelClassification(nn.Module):
    hparams={
        'max_len': 256,
        'model_name': 'bert-base-chinese'
    }
    def __init__(self, hparams: dict = None):
        """

        :param hparams:
            - num_labels
            - lr
            - *max_len
            - *model_name
        """
        super(MutilLabelClassification,self).__init__()
        self.hparams.update(hparams)
        self.max_len = self.hparams['max_len']
        self.bert = BertModel.from_pretrained(self.hparams['model_name'])
        self.tokenizer = BertTokenizerFast.from_pretrained(self.hparams['model_name'])
        self.bert.resize_token_embeddings(len(self.tokenizer))
        config = self.bert.config
        self.classifier = nn.Linear(config.hidden_size, self.hparams['num_labels'])
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True)
        #采用最后一层
        embedding = output.hidden_states[-1]
        embedding = self.pooling(embedding, attention_mask)
        output = self.classifier(embedding)
        return output

    def pooling(self,token_embeddings, attention_mask):
        output_vectors = []
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #这里做矩阵点积，就是对元素相乘(序列中padding字符，通过乘以0给去掉了)[B,L,768]
        t = token_embeddings * input_mask_expanded
        #[B,768]
        sum_embeddings = torch.sum(t, 1)

        # [B,768],最大值为seq_len
        sum_mask = input_mask_expanded.sum(1)
        #限定每个元素的最小值是1e-9，保证分母不为0
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        #得到最后的具体embedding的每一个维度的值——元素相除
        output_vectors.append(sum_embeddings / sum_mask)

        #列拼接
        output_vector = torch.cat(output_vectors, 1)

        return output_vector


    def encoding(self,inputs):
        self.bert.eval()
        with torch.no_grad():
            output = self.bert(**inputs, return_dict=True, output_hidden_states=True)
            embedding = output.hidden_states[-1]
            embedding = self.pooling(embedding, inputs)
        return embedding

    def get_optimizers(self):
        no_decay= ['bias', 'LayerNorm.weight']
        optimizer = AdamW([
            {
                'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.hparams['lr'],
                'weight_decay': self.hparams['weight_decay']
            }, {
                'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.hparams['lr'],
                'weight_decay': 0.0
            }
        ], lr=self.hparams['lr'])
        return optimizer


def multilabel_crossentropy(output,label):
    """
    多标签分类的交叉熵
    说明：label和output的shape一致，label的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证output的值域是全体实数，换言之一般情况下output
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出output大于0的类。如有疑问，请仔细阅读并理解
         本文。
    :param output: [B,C]
    :param label:  [B,C]
    :return:
    """
    output = (1-2*label)*output

    #得分变为负1e12
    output_neg = output - label* 1e12
    output_pos = output-(1-label)* 1e12

    zeros = torch.zeros_like(output[:,:1])

    # [B, C + 1]
    output_neg = torch.cat([output_neg,zeros],dim=1)
    # [B, C + 1]
    output_pos = torch.cat([output_pos,zeros],dim=1)


    loss_pos = torch.logsumexp(output_pos,dim=1)
    loss_neg = torch.logsumexp(output_neg,dim=1)
    loss = (loss_neg + loss_pos).sum()

    return loss


class BertMLC_FineTuner(pl.LightningModule):
    def __init__(self, params=None):
        super().__init__()
        self.hparams.update({
            'threshold': 0.5
        })
        if params is not None:
            self.hparams.update(params)
        self.model = MutilLabelClassification(params)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.hparams['model_name'])

    def training_step(self, batch, batch_idx):
        inp, tgt = batch

        output = self.model(
            input_ids=inp['input_ids'],
            token_type_ids=inp['token_type_ids'],
            attention_mask=inp['attention_mask']
        )  # (bsz, num_labels)
        target = tgt['label']  # (bsz, num_labels)
        loss = multilabel_crossentropy(output, target)
        self.log('loss', float(loss))
        return loss

    def train_dataloader(self):
        train_dataset = RM_Dataset('train', self.tokenizer, q=self.hparams['resample_q'])
        dataloader = DataLoader(train_dataset, batch_size=self.hparams['train_batch_size'], drop_last=True,
                                shuffle=True, collate_fn=RM_collate_fn)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def validation_step(self, batch, batch_idx):
        inp, tgt = batch

        output = self.model(
            input_ids=inp['input_ids'],
            token_type_ids=inp['token_type_ids'],
            attention_mask=inp['attention_mask']
        )

        # output = torch.sigmoid(output)
        pred = torch.where(output > 0, 1, 0)  # (bsz, num_labels)
        gt = tgt['label']
        return {
            'pred': pred,
            'gt': gt  # (bsz, num_labels)
        }

    def val_dataloader(self):
        train_dataset = RM_Dataset('valid', self.tokenizer)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams['eval_batch_size'], shuffle=False,
                                collate_fn=RM_collate_fn)
        return dataloader

    def validation_epoch_end(self, outputs):
        preds, gts = [], []
        for e in outputs:
            preds.append(e['pred'])
            gts.append(e['gt'])
        pred = torch.concat(preds, 0)  # (n, num_labels)
        gt = torch.concat(gts, 0)  # (n, num_labels)
        total, predict, correct = float(gt.sum()), float(pred.sum()), float((pred * gt).sum())
        precision = correct / predict if predict != 0 else 0
        recall = correct / total if total != 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0
        self.log_dict({
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        inp, tgt = batch
        texts, labels = inp['texts'], inp['labels']
        output = self.model(
            input_ids=inp['input_ids'],
            token_type_ids=inp['token_type_ids'],
            attention_mask=inp['attention_mask']
        )
        preds = []
        for e, t, l in zip(output, texts, labels):
            pred = (torch.where(torch.sigmoid(e) > 0))[0].tolist()
            real_labels = []
            for el in pred:
                real_labels.append(RM_labels[el])
            preds.append({
                'pred': real_labels,
                'text': t,
                'label': l
            })
        return preds

    def predict_dataloader(self):
        train_dataset = RM_Dataset('valid', self.tokenizer)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams['eval_batch_size'], shuffle=False,
                                collate_fn=RM_collate_fn)
        return dataloader

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def configure_optimizers(self):
        optimizer = self.model.get_optimizers()
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None,
                       optimizer_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None
                       ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()
