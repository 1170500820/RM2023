import sys
sys.path.append('..')

import random
import numpy as np
from loguru import logger as ru_logger
from argparse import ArgumentParser
import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from settings import *
from models.raw_bert import BertMLC_FineTuner


my_theme = RichProgressBarTheme(
    description="dark",
    progress_bar="green1",
    progress_bar_finished="green1",
    progress_bar_pulse="#6206E0",
    batch_progress="#ff1b66",
    time="cyan",
    processing_speed="cyan",
    metrics="black",
)

def handle_cli():
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, default='bert', help='使用的模型')
    parser.add_argument('--bsz', type=int, default=32, help='单卡的batch size。实际的batch size为bsz * n_gpus * grad_acc')
    parser.add_argument('--n_gpus', type=int, default=1, help='用于训练的显卡数量')
    parser.add_argument('--epoch', type=int, default=15, help='训练的epoch数')
    parser.add_argument('--name', type=str, default='RM_default', help='用于标识该次训练的名字，将用于对checkpoint进行命名。')
    parser.add_argument('--grad_acc', type=int, default=2, help='梯度累积操作，可用于倍增batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='模型使用的学习率')
    parser.add_argument('--seed', type=int, default=env_conf['seed'])
    # validate
    parser.add_argument('--val_interval', type=float, default=0.5, help='validate的间隔')
    # logger
    parser.add_argument('--logger_dir', type=str, default=logger_conf['logger_dir'])
    parser.add_argument('--every_n_epochs', type=int, default=logger_conf['every_n_epochs'])
    # checkpoint
    parser.add_argument('--ckp_dir', type=str, default=ckp_conf['dirpath'])
    parser.add_argument('--save_top_k', type=int, default=ckp_conf['save_top_k'])
    parser.add_argument('--monitor', type=str, default='f1_score')
    # model
    parser.add_argument('--model_name', type=str, default=plm_model_conf['model_name'])
    parser.add_argument('--max_length', type=int, default=plm_model_conf['max_seq_length'])
    parser.add_argument('--q', type=float, default=1)
    parser.add_argument('--disable_validation', action='store_true')

    # 在2个batch上进行过拟合实验
    parser.add_argument('--overfit', action='store_true', help='在2个batch上进行过拟合实验。验证代码正确性')

    args = vars(parser.parse_args())
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(config):
    return TensorBoardLogger(
        save_dir=config['logger_dir'],
        name=config['name']
    )


def get_callbacks(config):
    return [
        ModelCheckpoint(
            dirpath=config['ckp_dir'],
            save_top_k=config['save_top_k'],
            filename=config['name'] + '.' + '{epoch}-{f1_score:.3f}',
            monitor='f1_score',
            mode='max'),
        RichProgressBar(
            theme=my_theme)
    ]


def train(config):
    logger = get_logger(config)
    callbacks = get_callbacks(config)

    if config['n_gpus'] == 1:
        train_params = dict(
            accumulate_grad_batches=config['grad_acc'],
            accelerator=env_conf['accelerator'],
            max_epochs=config['epoch'],
            precision=32,
            logger=logger,
            callbacks=callbacks
        )
    else:  # config['n_gpus'] > 1

        train_params = dict(
            accumulate_grad_batches=config['grad_acc'],
            accelerator=env_conf['accelerator'],
            devices=config['n_gpus'],
            max_epochs=config['epoch'],
            strategy=env_conf['strategy'],
            precision=32,
            logger=logger,
            callbacks=callbacks
        )
    if config['overfit']:
        train_params['overfit_batches'] = 2
    train_params.update({
        'val_check_interval': config['val_interval']
        })
    model_params = dict(
        weight_decay=train_conf['weight_decay'],
        model_name=config['model_name'],
        lr=config['lr'],
        adam_epsilon=train_conf['adam_epsilon'],
        max_seq_length=plm_model_conf['max_seq_length'],
        warmup_steps=train_conf['warmup_steps'],
        train_batch_size=config['bsz'],
        n_gpus=config['n_gpus'],
        accumulate_grad_batches=config['grad_acc'],
        num_train_epochs=config['epoch'],
        eval_batch_size=config['bsz'],
        overfit=config['overfit'],

        num_labels=len(RM_labels),
        resample_q=config['q'],
        disable_validation=config['disable_validation']
    )

    ru_logger.info(f'正在加载模型{config["model_name"]}')
    if config['model'] == 'bert':
        model = BertMLC_FineTuner(model_params)
    else:
        raise Exception(f'未知的模型{config["model"]}')
    ru_logger.info('模型加载完毕')
    ru_logger.info('正在加载Trainer')
    trainer = pl.Trainer(**train_params)
    ru_logger.info('Trainer加载完毕，开始fit！')
    trainer.fit(model)


if __name__ == '__main__':
    conf = handle_cli()
    set_seed(conf['seed'])
    # logger.info(conf)
    train(conf)
