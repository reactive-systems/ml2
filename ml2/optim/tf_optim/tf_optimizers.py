"""Load TensorFlow optimizer"""

import copy

import tensorflow as tf

from ...registry import list_type_keys


def load_tf_optimizer_from_config(config: dict) -> tf.keras.optimizers.Optimizer:
    if "type" not in config:
        raise Exception("Optimizer type not specified in config")
    keras_config = copy.deepcopy(config)
    optimizer_type = keras_config.pop("type")
    if "name" not in keras_config:
        keras_config["name"] = optimizer_type
    if "learning_rate" in keras_config and isinstance(keras_config["learning_rate"], dict):
        learning_rate_config = keras_config["learning_rate"]
        if "type" not in learning_rate_config:
            raise Exception("Learning rate type not specified in config")
        learning_rate_type = learning_rate_config.pop("type")
        if "name" not in learning_rate_config:
            learning_rate_config["name"] = learning_rate_type
        if learning_rate_type in list_type_keys(
            bound=tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            learning_rate_type = "ml2>" + learning_rate_type
        keras_config["learning_rate"] = {
            "class_name": learning_rate_type,
            "config": learning_rate_config,
        }
    keras_serialization = {"class_name": optimizer_type, "config": keras_config}
    return tf.keras.optimizers.get(keras_serialization)


def tf_optimizer_to_config(optimizer: tf.keras.optimizers.Optimizer) -> dict:
    keras_serialization = tf.keras.optimizers.serialize(optimizer)

    config = keras_serialization["config"]
    config["type"] = keras_serialization["class_name"]
    if "learning_rate" in config and isinstance(config["learning_rate"], dict):
        learning_rate_serialization = config["learning_rate"]
        learning_rate_config = learning_rate_serialization["config"]
        if learning_rate_serialization["class_name"].startswith("ml2>"):
            learning_rate_config["type"] = learning_rate_serialization["class_name"][4:]
        else:
            learning_rate_config["type"] = learning_rate_serialization["class_name"]
        config["learning_rate"] = learning_rate_config
    for k, v in config.items():
        if type(v) not in [bool, dict, float, int, str]:
            # v is numpy type that needs to be converted to native Python type
            config[k] = v.item()
    return config
