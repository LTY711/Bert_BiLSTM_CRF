# 基于Bert+BiLSTM_CRF的中文NER

#### 介绍
基于Bert-base-chinese预训练模型
使用BERT_BiLSTM+CRF优化、微调的中文NER（人名、地名、机构名）项目

**整体代码架构图**

**数据预处理**

- 数据集目录结构 ✅
```
├── data
│   ├── doccano  
│   │   ├── export.json # doccano导出的文件
│   │   └── train_doccano.txt # 经过处理的序列标注数据
│   ├── origin
│   │   ├── 测试语料.txt 
│   │   └── 训练语料.txt # 原始数据
│   ├── processed
│   │   ├── msra_test_bio.txt 
│   │   └── msra_train_bio.txt # 开源数据 经过序列标注的数据
│   ├── rmrb_data 
│   │   ├── dev.txt
│   │   ├── example.dev
│   │   ├── example.test
│   │   ├── example.train # 人民日报数据
│   │   ├── test.txt
│   │   └── train.txt # 处理后的 序列标注数据
│   ├── labels.json # 实体类型对应的中文
│   ├── tag2id.json # 实体类型转换为标签形式
│   ├── test.txt 
│   └── train.txt # 合并后经过标注的数据

```
### 训练数据

标签采用BIO规则

|  标签	   |   描述    | 
|:------:|:-------:|
|   O	   |   非实体   |
| B-PER	 | 人名实体开始  | 
| I-PER  | 	人名实体结束 | 
| B-ORG  | 	机构名开始  | 
| I-ORG	 |  机构名结束  | 
| B-LOC	 |  地名开始   | 
| I-LOC	 |  地名结束   | 


每个实体类型的训练示例数

- [微软亚洲研究院 数据集](https://github.com/bytetopia/nlp_datasets/tree/master/ner/msra)

| 数据集 |   句数    |   字符数   | LOC数  | ORG数  | PER数  |
|:---:|:-------:|:-------:|:-----:|:-----:|:-----:|
| 训练集 |  45000  | 2171573 | 36860 | 20584 | 17615 |
| 测试集 |  3442   | 172601  | 2886  | 1331  | 1973  |

- 人民日报 数据集

| 数据集 |   句数    |  字符数   |  LOC数   |  ORG数  |   PER数   |
|:---:|:-------:|:------:|:-------:|:------:|:--------:|
| 训练集 |  20864  | 979180 |  16571  |  9277  |   8144   |
| 测试集 |  4636   | 219197 |  3658   |  2185  |   1864   |


### 训练过程

该模型在单个 NVIDIA A100 GPU 上进行训练

### 评估结果

|  metric	   | dev	  | test  |
|:----------:|:-----:|:-----:|
|    f1	     | 95.1  | 	91.3 |
| precision	 | 95.0	 | 90.7  |
|  recall	   | 95.3  | 	91.9 |
