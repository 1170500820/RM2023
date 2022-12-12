import sys
sys.path.append('..')

from loguru import logger
from argparse import ArgumentParser
from typing import Sequence, Any, Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from models.raw_bert import BertMLC_FineTuner
from datasets import RM_Dataset
from utils.io_tools import dump_jsonl
from settings import RM_labels, RM_labels_idx


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

    parser.add_argument('--model', type=str, choices=['bert'], default='bert')
    parser.add_argument('--ckp_file', type=str)
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--model_name', type=str, default='bert-base-chinese')

    args = vars(parser.parse_args())

    conf = dict(
        # 需要读取的ckp文件，以及需要预测的数据
        checkpoint_path=args['ckp_file'],
        name=args['name'],

        # 随机种子
        seed=42,

        # 模型参数
        model=args['model'],
        model_name=args['model_name'],
        max_seq_length=256,
        learning_rate=3e-4,
        weight_decay=0.1,
        adam_epsilon=1e-3,
        warmup_steps=0,

        # 训练参数
        train_batch_size=args['bsz'],
        eval_batch_size=args['bsz'],
        max_epochs=1,
        n_gpus=1,
        accumulate_grad_batches=4,
        strategy='ddp',
        accelerator='gpu',

        # 日志控制
        logger_dir='tb_log/',
        dirpath='RM-checkpoints/',
        every_n_epochs=1,
    )
    return conf


class PredWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval: str='epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        real_results = []
        for e in predictions[0]:
            real_results.extend(e)
        dump_jsonl(real_results, self.output_dir)


def get_callbacks(config):
    return [
        RichProgressBar(
            theme=my_theme),
        PredWriter(config['name'])
    ]


def predict(config):
    logger.info(f'使用的checkpoint文件：{config["checkpoint_path"]}')
    model_params = dict(
        weight_decay=config['weight_decay'],
        model_name=config['model_name'],
        lr=config['learning_rate'],
        adam_epsilon=config['adam_epsilon'],
        max_seq_length=config['max_seq_length'],
        warmup_steps=config['warmup_steps'],
        train_batch_size=config['train_batch_size'],
        n_gpus=config['n_gpus'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        num_train_epochs=config['max_epochs'],
        eval_batch_size=config['eval_batch_size'],
        num_labels=len(RM_labels)
    )
    if config['model'] == 'bert':
        model = BertMLC_FineTuner.load_from_checkpoint(config['checkpoint_path'], params=model_params)
    else:
        raise Exception(f'未知的模型:{config["model"]}')
    logger.info(f'{config["model"]}模型加载完成')

    # dataset = RM_Dataset(data_type='a', tokenizer=model.tokenizer)
    # loader = DataLoader(dataset, batch_size=config['eval_batch_size'])
    trainer = Trainer(callbacks=get_callbacks(config), gpus=1)

    model.model.eval()
    outputs = []
    model.model.to('cuda')
    logger.info('模型已迁移到CUDA，开始预测')
    trainer.predict(model)


if __name__ == '__main__':
    logger.info('预测流程已启动')
    conf = handle_cli()

    logger.info('参数读取完成，开始预测')
    predict(conf)

    logger.info('预测结束')