# coding=utf-8
# 用于向vocab和matrix中添加新的元素
import os.path
import pickle

import numpy as np
from gensim.models import Word2Vec


def expand_vocabulary(model_path, save_path):
    model = Word2Vec.load(model_path)
    id2vec = model.wv.vectors.tolist()
    id2word = model.wv.index2word
    vocab = ["<PAD>", "<S>", "<E>"]
    if "<UNK>" not in id2word:
        vocab.append("<UNK>")

    vectors = []
    for i in range(len(vocab)):
        vectors.append(list(np.random.uniform(low=-0.1, high=0.1, size=(300,))))

    vocab = vocab + id2word
    vectors = vectors + id2vec

    vocab_path = os.path.join(save_path, "vocab.pkl")
    vector_path = os.path.join(save_path, "vector.pkl")
    print(f"vocab_size {len(vocab)}")
    pickle.dump(vocab, open(vocab_path, "wb"))
    pickle.dump(vectors, open(vector_path, "wb"))


if __name__ == "__main__":
    model_path = "../checkpoints/word2vec/word2vec.model"
    save_path = "../checkpoints/data2text"
    expand_vocabulary(model_path, save_path)
