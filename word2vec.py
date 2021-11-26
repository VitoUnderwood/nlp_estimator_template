# coding=utf-8
# word2vec只针对特定的语料，不针对特定的任务
import argparse
import logging
import os.path

from gensim.models import Word2Vec

from utils.dir_op import create_dirs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MyCorpus:
    """ 语料每一行只有一句话，每句话分好词后用空格分隔开 """

    def __init__(self, corpus_path: str = None):
        self.corpus_path = corpus_path

    def __iter__(self):
        for line in open(self.corpus_path):
            yield [word for word in line.strip().split(" ") if len(word) > 0]


def train(args):
    create_dirs([args.output_dir])

    sentences = MyCorpus(args.corpus)
    model = Word2Vec(sentences=sentences,
                     size=args.vector_size,
                     min_count=args.min_count)

    # model.save(f"{args.output_dir}word2vec_{args.vector_size}.model")
    model = Word2Vec.load("checkpoints/word2vec/word2vec.model")
    model.wv.index2word()
    model.save(os.path.join(args.output_dir, args.word2vec_name))
    # model.wv.save_word2vec_format(args.output_dir + args.wordvec_name)


def get_args():
    parser = argparse.ArgumentParser(description="配置word2vec训练参数")
    parser.add_argument("--corpus", type=str, default="data/word2vec/corpus.txt", help="训练语料")
    parser.add_argument("--output_dir", type=str, default="checkpoints/word2vec", help="训练结果保存路径")
    parser.add_argument("--word2vec_name", type=str, default="word2vec.model",
                        help="specify the filename of the word vector")
    # Training Parameters
    parser.add_argument("--min_count", type=int, default=3, help="最少出现次数")
    parser.add_argument("--vector_size", type=int, default=300, help="词向量的维度")

    return parser.parse_args()


if __name__ == "__main__":
    config = get_args()
    train(config)
    # model = Word2Vec.load("word2vec.model")
    # # numpy vector of a word
    # vector = model.wv['computer']
