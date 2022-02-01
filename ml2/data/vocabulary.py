"""Vocabulary"""

import json
import logging
import os.path as path

from typing import Iterable


class Vocabulary:
    def __init__(self, token_to_id: dict, id_to_token: dict = None):
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
        logging.info(f"Constructed vocabulary containing {len(self.token_to_id)} tokens")

    def tokens_to_ids(self, tokens):
        return list(map(self.token_to_id.get, tokens))

    def ids_to_tokens(self, ids):
        return list(map(self.id_to_token.get, ids))

    def size(self):
        return len(self.token_to_id)

    def to_file(self, dir: str, filename: str):
        """
        Writes the vocabulary to a vocabulary file
        Args:
            dir: string, directory where vocabulary file is written
            filename: string, name of vocabulary file without filename extension
        """
        filepath = path.join(dir, filename + ".json")
        with open(filepath, "w") as vocab_file:
            json.dump(self.token_to_id, vocab_file, indent=2)
        logging.info(f"Written vocabulary to file {filepath}")

    @classmethod
    def from_file(cls, dir: str, filename: str):
        """
        Constructs a vocabulary from a vocabulary file
        Args:
            dir: string, directory that contains vocabulary file
            filename: string, name of vocabulary file without filename extension
        Returns:
            vocabulary object or None if vocabulary file can not be found or opened
        """
        filepath = path.join(dir, filename + ".json")
        if not path.isfile(filepath):
            logging.info(f"Could not find vocabulary file {filepath}")
            return None
        with open(filepath, "r") as vocab_file:
            token_to_id = json.load(vocab_file)
        return cls(token_to_id)

    @classmethod
    def from_iterable(cls, tokens: Iterable):
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
        return cls(token_to_id, id_to_token)
