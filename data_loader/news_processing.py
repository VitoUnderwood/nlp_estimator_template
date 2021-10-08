# -*- coding:utf-8 -*-
"""
将文件同一处理成csv文件或者tsv文件
title   label
"""
import os
import sys
sys.path.append("/Users/vito/PyCharmProjects/nlp_estimator_template")
from utils import dir_op
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold


def main():
    root_dir = "../data/THUCNews"
    avg_size = 3500
    proportion = {'train': 0.9,
                  'dev': 0.05,
                  'test': 0.05}
    print(proportion)

    assert proportion['train'] + proportion['dev'] + proportion['test'] == 1.0, "比例错误，sum应该为1"

    cat_list = dir_op.listdir_without_hidden_dir(root_dir)

    for cat in cat_list:
        file_list = dir_op.listdir_without_hidden_dir(os.path.join(root_dir, cat))
        print(cat, '\t', len(file_list))
    df = pd.DataFrame(columns=["title", "label"])
    for cat in tqdm(cat_list, desc="新闻类目", colour="green"):
        file_list = dir_op.listdir_without_hidden_dir(os.path.join(root_dir, cat))
        file_list = file_list[:avg_size]
        for file_name in tqdm(file_list, desc="新闻正文", colour="red", leave=False):
            with open(os.path.join(os.path.join(root_dir, cat), file_name), 'r') as f:
                title = f.readline().strip()
                label = cat
                df = df.append([{"title": title, "label": label}], ignore_index=True)

    print(f"total {len(df)} examples")

    train_df, test_dev_df = train_test_split(df, train_size=proportion['train'])
    dev_df, test_df = train_test_split(test_dev_df, train_size=proportion['dev']/(1-proportion['train']))

    print(f"train size {len(train_df)}")
    print(f"dev size {len(dev_df)}")
    print(f"test size {len(test_df)}")

    save_dir = "../data/News/"
    dir_op.create_dirs([save_dir])

    train_df.to_csv(os.path.join(save_dir, "train.tsv"), sep='\t')
    dev_df.to_csv(os.path.join(save_dir, "dev.tsv"), sep='\t')
    test_df.to_csv(os.path.join(save_dir, "test.tsv"), sep='\t')
    # kf = KFold(shuffle=True, random_state=2021)
    # for train_index, test_index in kf.split(df):
    #     train_df = df.iloc(train_index)
    #     test_df = df.iloc(test_index)


if __name__ == "__main__":
    main()
