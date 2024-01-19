"""Abstract class that can be constructed from and exported to a config"""

import copy
import inspect
import json
import logging
import os
from enum import Enum
from inspect import Parameter
from typing import Any, Dict, TypeVar, Union, get_args, get_origin

from .dtypes import DType
from .utils.typing_utils import is_subclass_generic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_no_annotation(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
    for name, value in config.items():
        if name not in annotations:
            logger.warning(f"Argument {name} has no type annotation.")


class Configurable:
    def get_config(self, **kwargs) -> dict:
        annotations = {}
        config = {}
        for name, param in self.init_params().items():
            if param.annotation is not param.empty:
                annotations[name] = param.annotation
            if name in config:
                continue
            if hasattr(self, name):
                config[name] = getattr(self, name)
            elif hasattr(self, "_" + name):
                config[name] = getattr(self, "_" + name)
            elif param.default is not Parameter.empty:
                config[name] = param.default
            # else:
            #     logger.warning(f"Could not determine config value of {name}")

        for postprocessor in self.config_postprocessors():
            postprocessor(config, annotations)

        return config

    def config_postprocessors(self) -> list:
        def postprocess_enum(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            for name, value in config.items():
                if (
                    name in annotations
                    and is_subclass_generic(annotations[name], Enum)
                    and isinstance(value, Enum)
                ):
                    config[name] = getattr(value, "value")

        def postprocess_types(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            for name, value in config.items():
                if (
                    name in annotations
                    and get_origin(annotations[name]) is type
                    and isinstance(value, type)
                ):
                    config[name] = value.__name__

        def postprocess_composed_types(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            # for example List[CSVDataset] or Dict[str, CSVDataset]
            # recurse preprocessing on Dict type
            for name, value in config.items():
                if (
                    isinstance(value, dict)
                    and name in annotations
                    and get_origin(annotations[name]) is dict
                ):
                    type_args = get_args(annotations[name])
                    assert len(type_args) == 2
                    child_config = {}
                    child_annotations = {}
                    for i, (k, v) in enumerate(value.items()):
                        child_config[f"k_{i}"] = k
                        child_config[f"v_{i}"] = v
                        child_annotations[f"k_{i}"] = type_args[0]
                        child_annotations[f"v_{i}"] = type_args[1]
                    for postprocessor in self.config_postprocessors():
                        postprocessor(child_config, child_annotations)
                    processed_dict = {}
                    for i, k in enumerate(list(value)):
                        processed_key = child_config[f"k_{i}"]
                        processed_value = child_config[f"v_{i}"]
                        processed_dict[processed_key] = processed_value
                    config[name] = processed_dict

        def postprocess_nested_artifact(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            for name, value in config.items():
                if (
                    name in annotations
                    and is_subclass_generic(annotations[name], Artifact)
                    and isinstance(value, Artifact)
                ):
                    config[name] = value.history if value.history else value.full_name

        def postprocess_nested_config(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            for name, value in config.items():
                if (
                    name in annotations
                    and is_subclass_generic(annotations[name], Configurable)
                    and isinstance(value, Configurable)
                ):
                    config[name] = value.get_config()

        from .artifact import Artifact

        def postprocess_class_type(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "type" not in config:
                config["type"] = self.__class__.__name__

        return [
            postprocess_enum,
            postprocess_types,
            postprocess_composed_types,
            postprocess_nested_artifact,
            postprocess_nested_config,
            process_no_annotation,
            postprocess_class_type,
        ]

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_basic_types(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            for name, value in config.items():
                if (
                    value is not None
                    and name in annotations
                    and annotations[name] in [int, float, str]
                ):
                    config[name] = annotations[name](value)

        def preprocess_enum(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            for name, value in config.items():
                if (
                    isinstance(value, str)
                    and name in annotations
                    and is_subclass_generic(annotations[name], Enum)
                ):
                    config[name] = annotations[name](value)

        def preprocess_underspecified_type(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            for name, value in config.items():
                if (
                    value is not None
                    and name in annotations
                    and annotations[name] in [dict, list, tuple]
                ):
                    logger.warning(
                        f"Type annotation {annotations[name]} of argument {name} is underspecified"
                    )
                    config[name] = annotations[name](value)

        from .registry import type_from_str

        def preprocess_ml2_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            # for example value 'LTLFormula' with annotation Type[BinaryExpr]
            for name, value in config.items():
                if (
                    value is not None
                    and name in annotations
                    and get_origin(annotations[name]) is type
                ):
                    type_args = get_args(annotations[name])
                    assert len(type_args) == 1
                    if is_subclass_generic(type_args[0], DType) and isinstance(value, str):
                        config[name] = type_from_str(value, bound=DType)

        def preprocess_composed_types(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            # for example List[CSVDataset] or Dict[str, CSVDataset]
            # recurse preprocessing on Dict type
            for name, value in config.items():
                if (
                    isinstance(value, dict)
                    and name in annotations
                    and get_origin(annotations[name]) is dict
                ):
                    type_args = get_args(annotations[name])
                    assert len(type_args) == 2
                    child_config = {}
                    child_annotations = {}
                    for i, (k, v) in enumerate(value.items()):
                        child_config[f"k_{i}"] = k
                        child_config[f"v_{i}"] = v
                        child_annotations[f"k_{i}"] = type_args[0]
                        child_annotations[f"v_{i}"] = type_args[1]
                    for preprocessor in Configurable.config_preprocessors():
                        preprocessor(child_config, child_annotations)
                    for i, k in enumerate(list(value)):
                        preprocessed_key = child_config[f"k_{i}"]
                        preprocessed_value = child_config[f"v_{i}"]
                        value.pop(k)
                        value[preprocessed_key] = preprocessed_value

        from .registry import type_from_str

        def preprocess_nested_config(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            for name, value in config.items():
                if (
                    (isinstance(value, dict) or isinstance(value, str))
                    and name in annotations
                    and is_subclass_generic(annotations[name], Configurable)
                ):
                    if isinstance(annotations[name], TypeVar):
                        annotations[name] = annotations[name].__bound__
                    if isinstance(value, dict) and "type" in value:
                        annotations[name] = type_from_str(value["type"], bound=Configurable)
                    if isinstance(value, str):
                        nested_config = annotations[name].fetch_config(value)
                        annotations[name] = type_from_str(
                            nested_config["type"], bound=Configurable
                        )
                    config[name] = annotations[name].from_config(value)

        def preprocess_config_type(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "type" in config:
                # pop such that type is not passed to init
                config_type = config.pop("type")
                if cls.__name__ != config_type:
                    logger.warning(
                        f"Config type {config_type} does not match class type {cls.__name__}"
                    )

        return [
            preprocess_basic_types,
            preprocess_enum,
            preprocess_underspecified_type,
            preprocess_ml2_dtype,
            preprocess_nested_config,
            preprocess_config_type,
            process_no_annotation,
            preprocess_composed_types,
        ]

    @classmethod
    def from_config(cls, config: Union[dict, str], **kwargs) -> "Configurable":
        if isinstance(config, str):
            artifact = cls.from_config_file(path=config)
            logger.info(f"Loaded {artifact.__class__.__name__} {artifact.name} from file {config}")
            return artifact

        config = copy.deepcopy(config)
        annotations = {}

        for name, param in cls.init_params().items():
            if name in kwargs:
                config[name] = kwargs[name]
            elif name not in config and param.default is not param.empty:
                config[name] = param.default
            if param.annotation is not param.empty:
                annotations[name] = param.annotation

        for preprocessor in cls.config_preprocessors():
            preprocessor(config, annotations)

        # check if all arguments without default are set
        # note that we not recurse over mro, which makes this an incomplete check
        for name, param in cls.init_params(recurse=False).items():
            if name not in config and param.default is param.empty:
                raise Exception(f"{name} not specified in {cls.__name__} config")

        return cls(**config)

    @classmethod
    def from_config_file(cls, path: str, **kwargs) -> "Configurable":
        if not os.path.exists(path):
            raise ValueError(f"Config path {path} does not exists locally")
        with open(path) as config_file:
            config = json.load(config_file)
        return cls.from_config(config=config, **kwargs)

    @classmethod
    def init_params(cls, recurse: bool = True) -> Dict[str, Parameter]:
        init_args = {}
        # fairly hacky - we assume that if kwargs is present it is passed to super constructor
        # thus we traverse mro until kwargs is not an argument
        for next_class in cls.mro() if recurse else [cls]:
            signature = inspect.signature(next_class.__init__)
            for name, parameter in signature.parameters.items():
                if name in ["self", "args", "kwargs"]:
                    continue
                if name not in init_args:
                    init_args[name] = parameter
            if "kwargs" not in signature.parameters:
                break
        return init_args

    @staticmethod
    def update_config_with_parent(config: dict, parent_config: dict) -> dict:
        # updates config with parent config recursively (without type configs)
        # the type key is used to avoid updates from the parent config
        # if "type" in config:
        #     return config
        for k, v in parent_config.items():
            if k not in config:
                config[k] = v
            elif isinstance(config[k], dict):
                if isinstance(parent_config[k], str):
                    config[k]["base"] = parent_config[k]
                else:
                    config[k] = Configurable.update_config_with_parent(config[k], parent_config[k])
            else:
                pass
        return config
