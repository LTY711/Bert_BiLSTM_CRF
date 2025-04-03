from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import time
import re
from config import *
from model.Bert_BiLSTM_CRF import *
from utils.common import build_data
from utils.data_loader import *
from flask import Flask, request, jsonify

app = Flask(__name__)
config = Config()

# datas = build_data(config.train_path)

# 实例化模型
models = {
    'Bert_BiLSTM_CRF': Bert_BiLSTM_CRF
}
model = models[config.model](config)

# strict 参数表示是否严格加载模型，如果为False，则只加载模型中包含的参数。
model.load_state_dict(torch.load('./save_model/Bert_BiLSTM_CRF_best.pth', map_location='cpu'))
id2tag = {value: key for key, value in config.tag2id.items()}

# TODO 测试集
def model2test(test_dataloader, model):
    aver_loss = 0
    preds, golds = [], []
    model.eval()  # 设置模型为评估模式
    # 自定义进度条
    pbar1 = tqdm(test_dataloader,
                 total=len(test_dataloader),
                 desc=f'{model.model_name}验证',
                 colour='blue')
    start_time = time.time()
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
    print(report)
    cost_time = int(time.time() - start_time)
    print(f'Bert_BiLSTM_CRF测试耗时：{cost_time / 60 :.2f}m')

    return precision, recall, f1, report

def extract_entities(tokens: list, labels: list):
    entities = []
    entity = []
    entity_type = None
    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            if entity:
                entities.append((entity_type    , ''.join(entity)))
                entity = []
            entity_type = label.split('-')[-1]
            entity.append(token)
        elif label.startswith('I-'): # 实体的中间或结尾
            entity.append(token)

        else:
            if entity:
                entities.append((entity_type, ''.join(entity)))
                entity = []
                entity_type = None
    if entity:
        entities.append((entity_type, ''.join(entity)))

    return {entity: entity_type for entity_type, entity in entities}

# TODO 单句模型预测
def model2test_sentence(sample: str):
    # 模型评估模式
    str_list = list(sample)
    tk = config.tokenizer.batch_encode_plus(
        [str_list],
        add_special_tokens=False,
        truncation=False,
        return_tensors='pt',
        is_split_into_words=True, # 如果是预分割的单词列表，输入是List[List[str]], 这里必须设为True
        return_attention_mask=True
    )

    # print(tk["input_ids"])
    res = model(tk["input_ids"], tk["attention_mask"])
    res_arr = []
    # print(f"res---->{res}")
    try:
        for i, word in enumerate(str_list):
            if res[0][i] == 0:
                continue

            res_arr.append((id2tag[res[0][i]], word))
    except Exception as e:
        print(f"数据有误：str_list---> {str_list}")


    print(res_arr)
    result_str = assemble_entities(res_arr)
    # print(f"result_str----->{result_str}")
    return result_str

# TODO 处理识别结果
def assemble_entities(tagged_sequence):

    per_str, loc_str, org_str = "", "", ""
    for tag, char in tagged_sequence:
        # print(f"tag--->{tag}")
        # print(f"char---->{char}")
        if tag.startswith('B-'):  # 新实体开始
            if "PER" in tag:
                per_str = per_str + "，" if len(per_str) > 0 else per_str
                per_str += char
            elif "LOC" in tag:
                loc_str = loc_str + "，" if len(loc_str) > 0 else loc_str
                loc_str += char
            elif "ORG" in tag:
                org_str = org_str + "，" if len(org_str) > 0 else org_str
                org_str += char

        elif tag.startswith('I-'):  # 同一实体延续
            if "PER" in tag:
                per_str += char
            elif "LOC" in tag:
                loc_str += char
            elif "ORG" in tag:
                org_str += char
        else:  # 非连续实体，结束当前实体
            print("上一步数据有问题")
            return []
    person = "人名：" + (per_str if len(per_str) > 0 else "无")
    local = "地名：" + (loc_str if len(loc_str) > 0 else "无")
    organize = "组织机构名：" + (org_str if len(org_str) > 0 else "无")
    return person+"\n"+local+"\n"+organize+"\n"

# TODO 封装识别API
@app.route("/service/api/bert_bilstm_crf", methods=['POST'])
def bert_bilstm_crf():
    data = {"sucess":0}
    param = request.get_json()
    sample = param["text"]
    sample = re.sub(r'\s+', '', sample)
    try:
        result = model2test_sentence(sample)
        data["result"] = result
        data["sucess"] = 1
    except:
        data["result"] = "嗯～抱歉，我不太明白你在说什么，让我自己学一会儿吧"
        data["sucess"] = 0

    return jsonify(data)



if __name__ == '__main__':
    # sample = '“投资中国就是投资未来” 习近平这句话含金量十足 推动高质量充分就业 健全社会保障体系 一个铸就辉煌仍勇于自我革命的党，才能无坚不摧'
    # sample = re.sub(r'\s+', '', sample)
    # model2test_sentence(sample)

    # test_dataloader = get_data()
    # model2test(test_dataloader, model)

    app.run(host='0.0.0.0', port=7009)

