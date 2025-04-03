import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from config import *
from utils.common import *

config = Config()
datas = build_data(config.train_path)
test_datas = build_data(config.test_path)


# 构建CustomDataset类
class CustomDataset(Dataset):
    """
    自定义CustomDataset类
    """

    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        x = self.datas[item][0] # 字
        y = self.datas[item][1] # 标注标签
        return x, y

# 构建自定义函数collate_fn()
def  collate_fn(batch):
    """
    自定义函数（将dataset中的data进行转张量等操作）

    :param batch:
    :return:
    """
    tokenizer = config.tokenizer
    # 拆分数据
    text_list = [data[0] for data in batch]
    labels_list = [data[1] for data in batch]
    # 3 分别组装，拿到目标值，并填充mask
    inputs = tokenizer.batch_encode_plus(
        text_list,
        padding='longest', # 动态填充当前批次的最大长度
        add_special_tokens=False,
        truncation=False,
        return_tensors='pt',
        is_split_into_words=True, # 如果是预分割的单词列表，输入是List[List[str]], 这里必须设为True,否则输入是List[str]，即每个样本是一个完整的字符串。
        return_attention_mask=True
    )
    # print(inputs['input_ids'][0])
    input_ids = inputs['input_ids'].to(config.device)
    attention_mask = inputs['attention_mask'].to(config.device)

    # 给labels转成张量
    std_length = input_ids.shape[1]
    label_list = []
    for label in labels_list:
        # print(label)
        if len(label) < std_length:
            # std_length - len(label) 即根据最长补齐
            label += ['O'] * (std_length - len(label))
            label_list.append(label)
        elif len(label) > std_length:
            label = label[:std_length]
            label_list.append(label)
        # TODO 不能遗忘等于时的情况
        else:
            label_list.append(label)


    # print(f"label_list->: {len(label_list)}")
    labels = torch.tensor([[config.tag2id[tag] for tag in labels] for labels in label_list])
    # print(f"labels-->{labels.shape}")
    labels = labels.to(config.device)

    return input_ids, labels, attention_mask


# 构建get_data函数，获得数据迭代器
def get_data():

    # data_len = len(datas)
    # print(f'数据集长度：{data_len}') # 65197
    # 分割数据集：8:2
    train_dataset = CustomDataset(datas[:52157])
    dev_dataset = CustomDataset(datas[52157:])
    test_dataset = CustomDataset(test_datas)
    # 处理数据集
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_fn,
                                  drop_last=True,
                                  shuffle=False)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_fn,
                                drop_last=True,
                                shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_fn,
                                drop_last=True,
                                shuffle=False)
    # return test_dataloader
    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    for i, (input_ids, labels, mask) in enumerate(train_dataloader):
        print(f"labels-->{labels.shape}")
        print(f"input_ids-->{input_ids.shape}")
        print(input_ids)
        break
    #     print(input_ids.shape)
    #     print(labels.shape)
    #     print(mask.shape)
    #     break
