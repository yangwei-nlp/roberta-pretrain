### Roberta预训练

#### 参考

代码基于[transformers](https://github.com/huggingface/transformers)，程序框架来源于[这里](https://huggingface.co/blog/how-to-train)。



#### 模型架构

预训练模型是`RobertaForMaskedLM`，也就是`RobertaModel`+`RobertaLMHead`两个部分组成；前者是RoBERTa的基础架构（**去掉了pooling层**），后者是掩码语言模型任务的架构。

```python
class RobertaForMaskedLM()
        def __init__(self, config):
            super().__init__(config)

            if not config.is_decoder:
                logger.warning("If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

            self.roberta = RobertaModel(config, add_pooling_layer=False)
            self.lm_head = RobertaLMHead(config)

            self.init_weights()
```



#### 训练语料

新浪新闻数据集（2005~2011年间），[来源](http://thuctc.thunlp.org/)。



#### 主要代码

**1.数据集`LineByLineTextDataset`**

代码介绍：

```python
# 1、将数据集复制10份，所以下面1个epoch等于10个epoch
self.examples = self.examples * 10

# 2、每篇新闻文本的512个字符作为一个train example
lines = []
cnt = 1
while cnt*512 <= length:
    start = cnt - 1
    lines.append(text[start*512: cnt*512])
    cnt += 1
lines.append(text[(cnt-1)*512: cnt*512])
```



**2.训练分词器**

基于当前数据集构造词典（当然也可以省略这个漫长的过程，ps，代码中没有自己训练分词器）



#### 模型使用

前面通过预训练得到了模型，如何在后续**对模型基于下游任务微调**？具体使用方法如下：

```python
from transformers import RobertaConfig, RobertaModel, BertTokenizer
config = RobertaConfig.from_json_file("data/checkpoint-100/config.json")
# 这里使用RobertaConfig而非BertConfig
roberta = RobertaModel.from_pretrained("data/checkpoint-100/pytorch_model.bin", config=config)
# 这里使用RobertaModel而非BertModel
```

注意，这里的RobertaModel是添加了**pooling**层的（为了适应下游任务），所以加载参数时会随机生成**pooling**层参数（两个参数）。

#### 注意，预训练使用什么(Bert/RoBERTa)模型，微调时就用什么(Bert/RoBERTa)模型的接口。



这里附三种模型加载方法：

```python
# 1.初始化参数的bert模型
raw_roberta = RobertaModel(config)  # 这里会随机初始化参数

# 2.自己确定参数名后加载参数
# raw_roberta.load_state_dict(new_state_dic)  # 由于参数命名不一致，会报错

# 3.hugging face出品，融合1,2
roberta = RobertaModel.from_pretrained("data/checkpoint-100/pytorch_model.bin", config=config)  # 加载了参数的roberta
```



