import torch
import torch.nn as nn
from TorchCRF import CRF
from transformers import BertTokenizer, BertConfig, BertModel
from config import *


class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, conf):
        super().__init__()

        # 1、加载Bert预训练模型
        self.bert = BertModel.from_pretrained(conf.bert_path, config=conf.bert_config)
        # 2、定义模型名称和参数
        self.model_name = 'Bert_BiLSTM_CRF'
        self.embedding_dim = conf.embedding_dim
        self.hidden_dim = conf.hidden_size
        self.tag_size = len(conf.tag2id)
        self.tag_to_ix = conf.tag2id
        self.dropout = nn.Dropout(conf.dropout)
        # 3、加载BiLSTM模型
        self.lstm = nn.LSTM(
            self.embedding_dim,
            hidden_size=self.hidden_dim // 2,
            bidirectional=True, # 双向
            batch_first=True
        )
        # 4、定义一个线性层
        self.linear = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.tag_size
        )
        # 5、定义CRF层
        self.crf = CRF(self.tag_size)

    # 算结果
    def forward(self, input_ids, mask):
        """
        预测（向前传播）
        句子经过字分词->字与词汇表id的映射表（list）->经过填充、切割、CLS等->转成张量
        :param x: 二维 input_ids(batch_size, sqlen)
        :param mask: 二维(input_ids, sqlen)
        :return:
        """
        # 获取Bert与LSTM的输出
        outputs = self.get_bert_lstm2linear(input_ids, mask)
        outputs = outputs * mask.unsqueeze(-1)
        return self.crf.viterbi_decode(outputs, mask)

    # 算损失(对数似然损失)
    def log_likelihood(self, x, tags, mask):
        outputs = self.get_bert_lstm2linear(x, mask)
        outputs = outputs * mask.unsqueeze(-1)
        # 计算损失
        # print("outputs shape:", outputs.shape)  # 应为 [16, seq_len, num_tags]
        # print("tags shape:", tags.shape)  # 应为 [16, seq_len]
        # print("mask shape:", mask.shape)  # 应为 [16, seq_len]
        return -self.crf(outputs, tags, mask)

    # 计算bert到BiLSTM的输出
    def get_bert_lstm2linear(self, input_ids, mask):
        # 获取Bert与LSTM的输出
        # Bert有2个输出，第一个是全部token（包含所有特殊标记）的三维的分类分数（通常为序列标注时使用），
        # 第二个是仅有[CLS]二维的对整句分类的一个分数（通常分类时使用）
        # print(f"input_ids--->: {input_ids.shape, input_ids}")
        outputs = self.bert(input_ids, mask)[0] # 因为是识别任务因此取0用第一个输出
        outputs, hidden = self.lstm(outputs)
        # 增加随机失活，防止过拟合
        outputs = self.dropout(outputs)
        return self.linear(outputs)


if __name__ == '__main__':
    import os
    import sys

    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(current_dir)
    # 将项目根目录添加到sys.path
    sys.path.append(project_root)
    print(project_root)
    from utils.common import *
    from utils.data_loader import *

    ner_bilstm_crf = Bert_BiLSTM_CRF(config)
    train_dataloader, dev_dataloader, test_dataloader = get_data()
    for x, y, mask in train_dataloader:
        mask = mask.to(torch.bool)
        result = ner_bilstm_crf.log_likelihood(x, y, mask)
        print(result)
        break
