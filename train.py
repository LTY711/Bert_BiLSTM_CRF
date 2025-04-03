import time

import torch
import torch.nn as nn
import torch.optim as optim
from model.Bert_BiLSTM_CRF import *
from utils.data_loader import *
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from config import *
from utils.data_loader import get_data

"""
编写训练函数
"""

# 实现模型训练函数
def model2train(train_dataloader, dev_dataloader, config):

    # 实例化模型
    model = Bert_BiLSTM_CRF(config)
    model = model.to(config.device)
    # 因为本次模型借助BERT做fine_tuning， 因此需要对模型中的大部分参数进行L2正则处理防止过拟合，包括权重w和偏置b
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]  # no_decay中存放不进行权重衰减的参数{因为bert官方代码对这三项免于正则化}
    # any()函数用于判断给定的可迭代参数iterable是否全部为False，则返回False，如果有一个为True，则返回True
    # 判断param_optimizer中所有的参数。如果不在no_decay中，则进行权重衰减;如果在no_decay中，则不进行权重衰减
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    # 优化器
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    # 是否需要对bert进行warm_up。这里默认不进行
    sheduler = None

    # 记录时间
    start_time = time.time()
    f1_score = -1000
    # 开始训练
    for epoch in range(config.epochs):
        model.train()
        # 自定义进度条
        pbar = tqdm(train_dataloader,
                    total=len(train_dataloader),
                    desc=f'Bert_BiLSTM_CRF训练[{epoch + 1}|{config.epochs}]',
                    colour='green')
        for idx, (inputs, labels, mask) in enumerate(pbar):
            # 将数据集放到GPU上去，如果有
            x = inputs.to(config.device)
            mask = mask.to(torch.bool).to(config.device)
            y = labels.to(config.device)
            # CRF 对数似然损失
            # print("-" * 30)
            # print(x.shape, mask.shape, y.shape)

            loss = model.log_likelihood(x, y, mask).mean()
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度裁剪 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
            # 参数更新
            optimizer.step()
            if idx % 200 == 0:
                pbar.set_postfix({
                    'loss': loss.item()
                })
        # 训练1个epoch后，进行验证
        precision, recall, f1, report = model2dev(dev_dataloader, model)
        if f1 > f1_score:
            f1_score = f1
            torch.save(model.state_dict(), f'save_model/{config.model}_best.pth')
            print(report)
        cost_time = int(time.time() - start_time)
        print(f'[{epoch + 1}|{config.epochs}]Bert_BiLSTM_CRF训练和验证耗时：{cost_time / 60 :.2f}m')

# 实现模型验证函数
def model2dev(dev_dataloader, model):
    aver_loss = 0
    preds, golds = [], []
    model.eval()  # 设置模型为评估模式
    # 自定义进度条
    pbar1 = tqdm(dev_dataloader,
                 total=len(dev_dataloader),
                 desc=f'{model.model_name}验证',
                 colour='blue')
    for idx, (inputs, labels, mask) in enumerate(pbar1):
        val_x = inputs.to(config.device)
        mask = mask.to(config.device)
        val_y = labels.to(config.device)
        mask = mask.to(torch.bool)

        predict = model(val_x, mask)
        my_loss = model.log_likelihood(val_x, val_y, mask).mean()
        aver_loss += my_loss.item()
        # 统计非0(真实标签长度)
        leng = []
        for i in val_x:
            tmp = []
            for j in i:
                if j != 0:
                    tmp.append(j.item())
            leng.append(tmp)
        # 取真实长度预测标签
        for idx, i in enumerate(predict):
            preds.extend(i[:len(leng[idx])])

        # 取真实长度真实标签
        for idx, i in enumerate(val_y.tolist()):
            golds.extend(i[:len(leng[idx])])

    # aver_loss /= (len(dev_iter) * 64)
    precision = precision_score(golds, preds, average='macro')
    recall = recall_score(golds, preds, average='macro')
    f1 = f1_score(golds, preds, average='macro')
    report = classification_report(golds, preds)
    return precision, recall, f1, report


if __name__ == '__main__':
    config = Config()
    train_dataloader, dev_dataloader, test_dataloader = get_data()

    model2train(train_dataloader, dev_dataloader, config)
