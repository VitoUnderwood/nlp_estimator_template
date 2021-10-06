import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.logger import Logger
from utils.utils import get_args


def main():
    # json配置文件读取以及命令行参数读取
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    # create tensorflow session
    sess = tf.Session()
    # 数据加载模块
    data = DataGenerator(config)
    # 模型实例化
    model = ExampleModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
