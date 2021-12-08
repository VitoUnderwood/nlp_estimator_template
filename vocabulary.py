# coding=utf-8
# 用于向vocab和matrix中添加新的元素
import pickle

import numpy as np
from gensim.models import Word2Vec

if __name__ == "__main__":
    word2vec_model = "checkpoints/word2vec/word2vec.model"
    model = Word2Vec.load(word2vec_model)
    id2vec = model.wv.vectors.tolist()
    id2word = model.wv.index2word
    vocab = ["<PAD>", "<S>", "<E>", "<UNK>"]
    vectors = []

    for i in range(len(vocab)):
        vectors.append(list(np.random.uniform(low=-0.1, high=0.1, size=(300,))))

    vocab = vocab + id2word
    vectors = vectors + id2vec

    # with open("checkpoints/word2vec/vocab.txt", "w") as f:
    #     for w in vocab:
    #         f.write(w+"\n")

    pickle.dump(vocab, open("checkpoints/QA_CVAE/vocab.pkl", "wb"))
    pickle.dump(vectors, open("checkpoints/QA_CVAE/vector.pkl", "wb"))
