"""Vocabulary"""

import json
import logging
import os
from typing import Any, Dict, Iterable, List

from ..artifact import Artifact
from ..registry import register_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class Vocabulary(Artifact):
    WANDB_TYPE = "vocabulary"

    def __init__(
        self,
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str] = None,
        filename: str = "vocab.json",
        name: str = "vocab",
        **kwargs,
    ):
        """
        Args:
            token_to_id: dict, mapping tokens to ids
            id_to_token, dict, optional, mapping ids to tokens
        """

        self.token_to_id = token_to_id
        if id_to_token:
            self.id_to_token = id_to_token
        else:
            self.id_to_token = {}
            for token, id in self.token_to_id.items():
                self.id_to_token[id] = token
        logger.info(f"Constructed vocabulary containing {len(self.token_to_id)} tokens")

        self.filename = filename

        super().__init__(name=name, **kwargs)

    def config_postprocessors(self) -> list:
        def postprocess_mappings(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("token_to_id", None)
            annotations.pop("token_to_id", None)
            config.pop("id_to_token", None)
            annotations.pop("id_to_token", None)

        return [
            postprocess_mappings,
        ] + super().config_postprocessors()

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return list(map(self.token_to_id.get, tokens))

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return list(map(self.id_to_token.get, ids))

    def size(self) -> int:
        return len(self.token_to_id)

    def save_to_path(self, path: str) -> None:
        """
        Writes the vocabulary to a vocabulary file
        Args:
            path: string, directory where vocabulary file is written
        """
        if self.filename is None:
            raise Exception("Filename not set")
        filepath = os.path.join(path, self.filename)
        with open(filepath, "w") as vocab_file:
            json.dump(self.token_to_id, vocab_file, indent=2)
        logger.info(f"Written vocabulary to file {filepath}")
        super().save_to_path(path)

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_vocab_file(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            path = cls.local_path_from_name(name=config["name"], project=config["project"])
            filepath = os.path.join(path, config["filename"])
            if "token_to_id" in config and config["token_to_id"] is not None:
                if os.path.isfile(filepath):
                    logger.warning("Vocabulary NOT loaded from existing file")
            else:
                if os.path.isfile(filepath):
                    with open(filepath, "r") as vocab_file:
                        token_to_id = json.load(vocab_file)
                        config["token_to_id"] = token_to_id
                else:
                    logger.warning("Constructing empty vocabulary")
                    config["token_to_id"] = {}

        return super().config_preprocessors() + [preprocess_vocab_file]

    @classmethod
    def from_iterable(
        cls,
        tokens: Iterable,
        filename: str = "vocab.json",
        name: str = "vocab",
        auto_version: bool = False,
        project: str = None,
    ) -> "Vocabulary":
        """
        Constructs a vocabulary from an iterable
        Args:
            tokens: iterable
        Returns:
            vocabulary object
        """
        token_to_id = {}
        id_to_token = {}
        for id, token in enumerate(tokens):
            token_to_id[token] = id
            id_to_token[id] = token

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            filename=filename,
            name=name,
            project=project,
            auto_version=auto_version,
        )
