# coding=utf-8
import os
import sys
import re
from utils import dir_op
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.append("/Users/vito/PyCharmProjects/nlp_estimator_template")


def clean_text(text):
    """Clean text by removing unnecessary characters and altering the format of words."""

    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"temme", "tell me", text)
    text = re.sub(r"gimme", "give me", text)
    text = re.sub(r"howz", "how is", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r" & ", " and ", text)
    text = re.sub(r"[-()\"#[\]/@;:<>{}`*_+=&~|.!/?,]", "", text)

    return text


def main():
    root_dir = "../../raw_data/NMT"
    proportion = {'train': 0.9,
                  'dev': 0.05,
                  'test': 0.05}
    print(proportion)

    assert proportion['train'] + proportion['dev'] + proportion['test'] == 1.0, "比例错误，sum应该为1"

    df = pd.DataFrame(columns=["eng", "deu"])

    vocab = []

    with open(root_dir + "/deu.txt", "r") as f:
        for line in tqdm(f.readlines(), desc="英文、德文句子对\n", colour="green"):
            eng, deu = line.strip().split('\t')
            eng = clean_text(eng)
            deu = clean_text(deu)
            eng_list = [word for word in eng.strip().split(" ") if len(word) > 0]
            for w in eng_list:
                if w not in vocab:
                    vocab.append(w)
            deu_list = [word for word in deu.strip().split(" ") if len(word) > 0]
            for w in deu_list:
                if w not in vocab:
                    vocab.append(w)
            df = df.append([{"eng": eng, "deu": deu}], ignore_index=True)

    print(f"total {len(df)} examples")

    train_df, test_dev_df = train_test_split(df, train_size=proportion['train'])
    dev_df, test_df = train_test_split(test_dev_df, train_size=proportion['dev'] / (1 - proportion['train']))

    print(f"train size {len(train_df)}")
    print(f"dev size {len(dev_df)}")
    print(f"test size {len(test_df)}")

    save_dir = "../data/NMT/"
    dir_op.create_dirs([save_dir])

    train_df.to_csv(os.path.join(save_dir, "train.tsv"), sep='\t')
    dev_df.to_csv(os.path.join(save_dir, "dev.tsv"), sep='\t')
    test_df.to_csv(os.path.join(save_dir, "test.tsv"), sep='\t')

    print(f"vocab size is {len(vocab)}")

    with open(os.path.join(save_dir, "vocab.txt"), "w") as f:
        vocab.insert(0, "[GO]")
        vocab.insert(0, "[EOS]")
        vocab.insert(0, "[UNK]")
        vocab.insert(0, "[PAD]")
        for w in vocab:
            f.write(w + "\n")


if __name__ == "__main__":
    main()
