import pandas as pd
import json
import os
import sys
from tqdm import tqdm

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录的绝对路径
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到sys.path
sys.path.append(project_root)


# doccano数据处理
def doccano_process(entities):
    """
    doccano导出的数据处理
    :param entities:
    :return:
    """
    tag2entity = {'Place': 'LOC', 'Person': 'PER', 'Organization': 'ORG'}
    res_dict = {}
    if len(entities) > 0:  # 判断是否有实体类型
        for entity in entities:
            label = entity['label']  # 标签名
            start_offset = entity['start_offset']  # 索引开始
            end_offset = entity['end_offset']  # 索引结束
            label_tag = tag2entity.get(label)
            for i in range(start_offset, end_offset):
                if i == start_offset:
                    tag = 'B-' + label_tag
                else:
                    tag = 'I-' + label_tag
                res_dict[i] = tag
    return res_dict


# doccano 序列标注并保存到文件
def doccano_write_train():
    with open(project_root + '/data/doccano/train_doccano.txt', 'w', encoding='utf-8') as fa:
        with open(project_root + '/data/doccano/export.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                obj = json.loads(line)
                entities = obj['entities']  # 实体列表
                res_dict = doccano_process(entities)
                text = obj['text']
                for idx, token in enumerate(text):
                    new_token = token + '\t' + res_dict.get(idx, 'O')
                    fa.write(new_token + '\n')


# 处理

# 处理人民日报数据格式
def rmrb_process():
    # 训练集
    with open(project_root + '/data/rmrb_data/example.train', 'r', encoding='utf-8') as file_in, open(
            project_root + '/data/rmrb_data/train.txt', 'w', encoding='utf-8') as file_out:
        for line in file_in:
            data = line.split()  # 使用空格分割,返回列表
            line_w_tabs = "\t".join(data) + "\n"
            file_out.write(line_w_tabs)
    # 测试集
    with open(project_root + '/data/rmrb_data/example.test', 'r', encoding='utf-8') as file_in, open(
            project_root + '/data/rmrb_data/test.txt', 'w', encoding='utf-8') as file_out:
        for line in file_in:
            data = line.split()  # 使用空格分割,返回列表
            line_w_tabs = "\t".join(data) + "\n"
            file_out.write(line_w_tabs)
    # 验证集
    with open(project_root + '/data/rmrb_data/example.dev', 'r', encoding='utf-8') as file_in, open(
            project_root + '/data/rmrb_data/dev.txt', 'w', encoding='utf-8') as file_out:
        for line in file_in:
            data = line.split()  # 使用空格分割,返回列表
            line_w_tabs = "\t".join(data) + "\n"
            file_out.write(line_w_tabs)

    print('处理人民日报数据格式完成')


# 合并多个文件
def merge_data():
    # 合并 训练集
    train_filenames = [
        # project_root + '/data/doccano/train_doccano.txt',
        project_root + '/data/processed/msra_train_bio.txt', #
        project_root + '/data/rmrb_data/train.txt'  #人民日报
    ]
    with open(project_root + '/data/train.txt', 'w', encoding='utf-8') as outfile:
        for fname in train_filenames:
            with open(fname) as infile:
                for line in infile:
                    if line.split('\t')[0] == '0':
                        continue
                    outfile.write(line)
    # 合并 测试集
    train_filenames = [
        # project_root + '/data/doccano/train_doccano.txt',
        project_root + '/data/processed/msra_test_bio.txt',
        project_root + '/data/rmrb_data/test.txt'
    ]
    with open(project_root + '/data/test.txt', 'w', encoding='utf-8') as outfile:
        for fname in train_filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    print('合并文件完成')


if __name__ == '__main__':
    # doccano_write_train() # doccano导出文件序列标注
    # rmrb_process() # 处理人民日报数据格式
    merge_data() # 合并数据
    # ...
