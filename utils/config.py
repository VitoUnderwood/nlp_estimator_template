import json
import os

from bunch import Bunch

from utils.utils import create_dirs


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    return config


def process_config(json_file):
    config = get_config_from_json(json_file)
    config.summary_dir = os.path.join("summary/")
    config.checkpoint_dir = os.path.join("checkpoints/")
    config.best_checkpoint_dir = os.path.join(config.checkpoint_dir, "best/")
    config.tmp_checkpoint_dir = os.path.join(config.checkpoint_dir, "tmp/")
    create_dirs([config.summary_dir,
                config.checkpoint_dir,
                config.best_checkpoint_dir,
                config.tmp_checkpoint_dir
                 ])
    return config
