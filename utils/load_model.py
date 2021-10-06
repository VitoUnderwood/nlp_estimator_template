# -*- coding:utf-8 -*-
import tensorflow_hub as hub
import tensorflow as tf

def load_model(hub_module_handle, is_training=True):
    tags = set()
    if is_training:
        tags.add("train")
    model = hub.Module(hub_module_handle, tags=tags, trainable=True)
    return model

def save_model(hub_module_handle):
    hub.saved_model_module()
    hub.load_module_spec()