"""
模型加载方法
"""

import torch
from transformers import RobertaConfig, RobertaModel, BertTokenizer

config = RobertaConfig.from_json_file("data/checkpoint-100/config.json")
# 切记这里使用RobertaConfig而非BertConfig，因为参数命名前缀不一致会导致参数加载失败

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")  # bert和roberta分词器一样，无所谓
roberta = RobertaModel.from_pretrained("data/checkpoint-100/pytorch_model.bin", config=config)  # 加载了参数的roberta
# 这里使用RobertaModel而非BertModel

"""
PS,
raw_roberta = RobertaModel(config)  # 这里会随机初始化参数
# raw_roberta.load_state_dict(new_state_dic)  # 由于参数命名不一致，会报错
"""
