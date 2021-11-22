# -*- coding:utf-8 -*-
import os


def listdir_without_hidden_dir(path):
    dir_list = os.listdir(path)
    without_hidden_dir_list = []
    for d in dir_list:
        if (d.startswith('.')):
            continue
        without_hidden_dir_list.append(d)
    return without_hidden_dir_list


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    assert isinstance(dirs, list), "请输入list"
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
