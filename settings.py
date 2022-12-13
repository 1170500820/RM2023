import json


data_path = 'data/filtered/processed_1021/'

RM_labels = json.load(open(data_path + 'label_list.json', 'r', encoding='utf-8'))
RM_labels_idx = json.load(open(data_path + 'label_idx.json', 'r', encoding='utf-8'))


"""
Environment -- 训练的基础信息，包括随机数种子、训练的名字（标识）
"""
env_conf = dict(
    # 随机数种子
    seed=42,
    # 本次训练的命名
    name='RM_default',
    # 其他
    n_gpus=1,
    accelerator='gpu',
    strategy='ddp'
)


"""
Train -- 训练流程的参数，指定了Trainer的行为
"""
train_conf = dict(
    # 基础训练参数
    learning_rate=1e-5,
    weight_decay=0.1,
    adam_epsilon=1e-3,
    warmup_steps=0,
    max_epochs=20,
    accumulate_grad_batches=2,
    # batch参数
    train_batch_size=32,
    eval_batch_size=16
)


"""
Model -- 模型参数，指定了Model、FineTuner的行为
"""
model_conf = dict()

plm_model_conf = dict(
    model_name='bert-base-chinese',
    max_seq_length=256,
)


"""
Logger -- 指定了Logger的行为
"""
logger_conf = dict(
    logger_dir='tb_log/',
    every_n_epochs=1,
)

ckp_conf = dict(
    save_top_k=-1,
    monitor='f1_score',
    dirpath='RM-checkpoints/',
)
