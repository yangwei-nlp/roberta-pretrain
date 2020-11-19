"""
预训练RoBERTa，使用清华大学整理的新闻语料库-THUCNews
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import configparser
from typing import Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizers import ByteLevelBPETokenizer


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        """
        处理所有文本语料，得到固定长度的字符串以供训练

        Args:
            tokenizer: 分词器，用来将文本转化为词id
            file_path: 训练语料的路径
            block_size: 字符串最大长度
        """
        self.examples = []
        t_0 = time.time()
        # f_paths = [str(x) for x in Path(file_path).glob("**/*.txt")][:100]  # 调试时可以打开这行代码，注释下行
        f_paths = [str(x) for x in Path(file_path).glob("**/*.txt")]
        for f in f_paths:
            with open(f, encoding="utf-8") as fp:
                text = fp.read()

            this_one_examples = self.cut_sent(text)
            
            lines = [item for item in this_one_examples if (len(item) > 10)]
            if not lines:
                continue
            batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
            if "input_ids" not in batch_encoding:
                print("Here...")
            tmp = batch_encoding["input_ids"]
            self.examples.extend([{"input_ids": torch.tensor(e, dtype=torch.long)} for e in tmp])
        self.examples = self.examples * 10  # same as roberta paper
        print(f"样本数: {len(self.examples)}, 耗时: {(time.time()-t_0)/60:.2f} mins")
    

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


    def cut_sent(self, text):
        lines = []
        text = text.replace("\u3000", "").replace("\xa0", "").replace("\n\n", "").replace("\n", "")
        length = len(text)
        cnt = 1
        while cnt*512 <= length:
            start = cnt - 1
            lines.append(text[start*512: cnt*512])
            cnt += 1
        lines.append(text[(cnt-1)*512: cnt*512])
        return lines


def train_tokenizer(paths, vocab_size=21128, min_frequency=2):
    """
    训练tokenizer，并保存到本地; 如果数据量大可能会很耗时.

    Args:
        paths: 训练用的文本文件目录
        vocab_size: 词典大小
        min_frequency: 出现次数小于该值的单词被过滤掉

    Returns：
        将词典保存到本地，返回分词器对象
    """
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    # Customize training
    tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    tokenizer.save_model("data/THUCBert")  # 保存分词器（其实就是个词典）

    return tokenizer


def train(epoch, vocab_size, train_files_path, save_path, learning_rate, save_steps, 
        per_gpu_train_batch_size, gradient_accumulation_steps):
    """
    从头训练一个语言模型RoBERTa.
    """
    from transformers import RobertaConfig

    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=1,
    )

    # 不训练，直接用BERT的分词器
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", max_len=512)
    # 事实上，hfl/chinese-roberta-wwm-ext/vocab.txt == chinese_L-12_H-768_A-12/vocab.txt

    from transformers import RobertaForMaskedLM
    """
    源码：
    class RobertaForMaskedLM()
        def __init__(self, config):
            super().__init__(config)

            if not config.is_decoder:
                logger.warning("If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

            self.roberta = RobertaModel(config, add_pooling_layer=False)
            self.lm_head = RobertaLMHead(config)

            self.init_weights()

    可以发现，RobertaForMaskedLM包含了roberta主体模型(不含pooling层)和一个语言模型输出层
    """
    # 初始化模型
    model = RobertaForMaskedLM(config=config)

    print("参数数量: ", model.num_parameters())  # 1亿参数

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_files_path,
        block_size=512,
    )

    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_gpu_train_batch_size=per_gpu_train_batch_size,
        save_steps=save_steps,
        save_total_limit=2,
        gradient_accumulation_steps=gradient_accumulation_steps,  # 32*8=256 batch_size
        learning_rate=learning_rate,
        weight_decay=0.01,
        adam_epsilon=1e-6,
        warmup_steps=10000,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model(save_path)


def check_model(save_path):
    """
    通过判断句子是否通顺，来判断模型是否成功训练
    """
    from transformers import pipeline

    fill_mask = pipeline(
        "fill-mask",
        model=save_path,
        tokenizer="bert-base-chinese"
    )
    print(fill_mask("我是你的[MASK]."))
    print(fill_mask("你是不是个傻[MASK]啊？."))


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Args
    epoch = config.getint('Parameters', 'epoch')
    vocab_size = config.getint('Parameters', 'vocab_size')
    train_files_path = config.get('Parameters', 'train_files_path')
    save_path = config.get('Parameters', 'save_path')
    learning_rate = config.getfloat('Parameters', 'learning_rate')
    save_steps = config.getint('Parameters', 'save_steps')
    per_gpu_train_batch_size = config.getint('Parameters', 'per_gpu_train_batch_size')
    gradient_accumulation_steps = config.getint('Parameters', 'gradient_accumulation_steps')

    train(
        epoch, vocab_size, train_files_path, save_path, learning_rate, save_steps, 
        per_gpu_train_batch_size, gradient_accumulation_steps
    )
    check_model(save_path)

    print("Over!!!")
    # nohup python -u pretrain.py > logs/pretrain-11-11.log 2>&1 &

