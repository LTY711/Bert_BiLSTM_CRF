import os
import torch
import json
from transformers import BertTokenizer, BertModel, BertConfig


current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径

class Config:
    def __init__(self):
        self.device = self._get_device()
        # 加载数据集路径
        self.bert_path = os.path.join(current_dir, "data", "bert_pretrain")
        self.train_path = os.path.join(current_dir, "data", "train.txt")
        self.test_path = os.path.join(current_dir, "data", "test.txt")
        self.vocab_path = os.path.join(current_dir, "data", "bert_pretrain", "vocab.txt")

        # 模型参数
        # vocab 改成自己的字典，输入为list
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(os.path.join(self.bert_path, "bert_config.json"))
        self.tag2id = json.load(open(os.path.join(current_dir, 'data', 'tag2id.json'), encoding='utf-8'))
        self.tag_size = len(self.tag2id)
        self.max_length = 150
        self.embedding_dim = 768
        self.hidden_size = 256
        self.batch_size = 32
        self.epochs = 30
        self.learning_rate = 1e-5
        self.dropout = 0.2
        self.model = 'Bert_BiLSTM_CRF'

    def _get_device(self):
        """
        检测并设置最佳可用的计算设备
        """
        if torch.cuda.is_available():
            return torch.device("cuda")  # gpu
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  #
        #     return torch.device("mps")  # mac m1 m2...
        else:
            return torch.device("cpu")


if __name__ == '__main__':
    config = Config()
    print(config.tag2id)
    print(config.tokenizer)
