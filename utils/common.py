from config import *
from tqdm import tqdm

"""
构建词表
"""
config = Config()


# 构造样本x以及标签y数据对
def build_data(path):
    datas = []
    sample_x = []
    sample_y = []
    with open(path, 'r', encoding='utf-8') as f:
        readlines = f.readlines()
        pbar = tqdm(readlines, total=len(readlines), colour='yellow', desc='构建数据中')
    # 遍历数据集，将每一行数据以\t分割，分别获取字符和标签，取完整句子和其对应的标注标签存储到list/tuple
    for line in pbar:
        if not line:
            continue
        line = line.strip().split('\t')
        char = line[0]
        if not char:
            continue
        cate = line[-1]
        sample_x.append(char)
        sample_y.append(cate)

        if char in ['。', '?', '!', '！', '？']:
            # 因为bert的512长度限制，所以在这里加入截断逻辑
            if len(sample_x) > 510:
                sample_x = sample_x[:510]
                sample_y = sample_y[:510]
            datas.append([sample_x, sample_y]) # 句子与标签对
            sample_x = []
            sample_y = []

    return datas


# 构造样本x以及标签y数据对
def get_data_info(file_path, dataset='数据集', data_type=''):
    datas, sample_x, sample_y = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n\n')
        pbar = tqdm(lines, total=len(lines), colour='yellow', desc='数据集信息获取中')
    for line in pbar:
        for value in line.split('\n'):
            word, tag = value.split('\t')
            sample_x.append(word)
            sample_y.append(tag)
        datas.append([sample_x, sample_y])
        sample_x = []
        sample_y = []
    print(f'\n ====> {dataset} <===')
    print(f'\n{data_type}句子数量:', len(datas))
    print(f'{data_type}Token数量:', sum([len(data[0]) for data in datas]))
    print('LOC数量:', sum([data[1].count("B-LOC") for data in datas]))
    print('ORG数量:', sum([data[1].count("B-ORG") for data in datas]))
    print('PER数量:', sum([data[1].count("B-PER") for data in datas]))
    print('-' * 80)
    return datas


if __name__ == '__main__':
    # get_data_info(config.msra_train_path, 'msra 数据集', '训练集')
    # get_data_info(config.msra_test_path, 'msra 数据集', '测试集')
    # get_data_info(config.rmrb_train_path, '人民日报 数据集', '训练集')
    # get_data_info(config.rmrb_test_path, '人民日报 数据集', '测试集')

    # get_data_info(config.train_path, '合并后 数据集', '训练集')
    # get_data_info(config.test_path, '合并后 数据集', '测试集')

    data = build_data()
    count = 0
    for sentence in data:
        if len(sentence[0]) > 510:
            count += 1
            print(f"sentence的长度大于510的： {sentence}")

    print(count)
    # print(data[2:])
