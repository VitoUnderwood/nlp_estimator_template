# coding=utf-8
import json
import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("../")

from utils import dir_op


def is_chinese_char(c):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.

    # 可以参考 https://www.qqxiuzi.cn/zh/hanzi-unicode-bianma.php
    # 获取字符的编码ord
    cp = ord(c)
    # 这里只判断中文字，但不会判断中文的符号
    if ((0x4E00 <= cp <= 0x9FFF) or
            (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or
            (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or
            (0x2F800 <= cp <= 0x2FA1F)):
        return True
    return False


def clean_text(text):
    """Clean text by removing unnecessary characters and altering the format of words.
    nlg 任务的时候尽量保持原有的句子结构，不要去除停用词和标点，否则直接影响流畅程度
    """

    text = re.sub(r"\s", "", text)

    return text


def main():
    root_dir = "../data/raw_data/baike2018qa/"
    proportion = {'train': 0.9,
                  'dev': 0.05,
                  'test': 0.05}
    print(proportion)

    assert proportion['train'] + proportion['dev'] + proportion['test'] == 1.0, "比例错误，sum应该为1"

    # {"qid":<qid>,"category":<category>,"title":<title>,"desc":<desc>,"answer":<answer>}
    #
    # 其中，category是问题的类型，title是问题的标题，desc是问题的描述，可以为空或与标题内容一致。

    df = pd.DataFrame(columns=["question", "answering"])

    with open(root_dir + "data.json", "r") as f:
        for line in tqdm(f.readlines(), desc="QA数据", colour="green"):
            tmp = json.loads(line)
            question = clean_text(tmp["title"])
            answering = clean_text(tmp["answer"])
            if len(question) > 0 and len(answering) > 0:
                df = df.append([{"question": question, "answering": answering}], ignore_index=True)
            else:
                print("blank line")

    print(f"total {len(df)} examples")

    train_df, test_dev_df = train_test_split(df, train_size=proportion['train'])
    dev_df, test_df = train_test_split(test_dev_df, train_size=proportion['dev'] / (1 - proportion['train']))

    print(f"train size {len(train_df)}")
    print(f"dev size {len(dev_df)}")
    print(f"test size {len(test_df)}")

    save_dir = "../data/QA_CVAE/"
    dir_op.create_dirs([save_dir])

    train_df.to_csv(os.path.join(save_dir, "train.tsv"), sep='\t')
    dev_df.to_csv(os.path.join(save_dir, "dev.tsv"), sep='\t')
    test_df.to_csv(os.path.join(save_dir, "test.tsv"), sep='\t')


if __name__ == "__main__":
    main()
