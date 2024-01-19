"""Abstract category sequence pair class"""

from abc import abstractmethod
from typing import Generic, List, TypeVar

from .cat import Cat
from .pair import GenericPair, Pair
from .seq import Seq

C = TypeVar("C", bound=Cat)
S = TypeVar("S", bound=Seq)


class CatSeq(Pair[C, S], Generic[C, S]):
    @property
    @abstractmethod
    def cat(self) -> C:
        raise NotImplementedError()

    @property
    def fst(self) -> C:
        return self.cat

    @property
    def snd(self) -> S:
        return self.seq

    @property
    @abstractmethod
    def seq(self) -> S:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_cat_seq_pair(cls, cat: C, seq: S, **kwargs) -> "CatSeq[C, S]":
        raise NotImplementedError()

    @classmethod
    def from_components(cls, fst: C, snd: S, **kwargs) -> "CatSeq[C, S]":
        return cls.from_cat_seq_pair(cat=fst, seq=snd)

    @classmethod
    @abstractmethod
    def from_cat_seq_tokens(
        cls, cat_token: str, seq_tokens: List[str], **kwargs
    ) -> "CatSeq[C, S]":
        raise NotImplementedError()


class GenericCatSeq(GenericPair[C, S], Generic[C, S]):
    @property
    def cat(self) -> C:
        return self.fst

    @property
    def seq(self) -> S:
        return self.snd

    @classmethod
    def from_cat_seq_pair(cls, cat: C, seq: S, **kwargs) -> "GenericCatSeq[C, S]":
        return cls(fst=cat, snd=seq)

    @classmethod
    @abstractmethod
    def from_cat_seq_tokens(
        cls, cat_token: str, seq_tokens: List[str], **kwargs
    ) -> "GenericCatSeq[C, S]":
        raise NotImplementedError()
