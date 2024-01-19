"""Tree class inspired by the nltk tree class https://www.nltk.org/_modules/nltk/tree.html"""

from enum import Enum
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class TraversalOrder(Enum):
    PRE = "pre"
    IN = "in"
    POST = "post"


class Tree(list, Generic[T]):
    def __init__(self, label: T, children: List["Tree[T]"]):
        list.__init__(self, children)
        self.label = label

    @property
    def leaves(self) -> List[T]:
        """Returns a list of (labels of) leaves"""
        if not self:
            return [self.label]
        leaves = []
        for child in self:
            leaves.extend(child.leaves)
        return leaves

    def rename(self, rename: Dict[str, str]):
        if not self:
            if self.label in rename.keys():
                self.label = rename[self.label]
        else:
            for child in self:
                child.rename(rename)

    @property
    def breadth(self) -> int:
        """Returns the breadth of the tree, i.e., the number of leaves"""
        return len(self.leaves)

    @property
    def degree(self) -> int:
        """Returns the degree of the tree, i.e., the maximum number of children a node has"""
        degree = len(self)
        for child in self:
            degree = max(degree, child.degree())
        return degree

    @property
    def height(self) -> int:
        """Returns the height of the tree, i.e., the length of the longest path from the root node to a leaf node"""
        max_child_height = -1
        for child in self:
            max_child_height = max(max_child_height, child.height)
        return max_child_height + 1

    def size(self, **kwargs) -> int:
        """Returns the size of the tree, i.e., the number of nodes"""
        size = 1
        for child in self:
            size += child.size(**kwargs)
        return size

    def traversal(
        self,
        order: TraversalOrder,
        partition_fn: Optional[
            Callable[[List["Tree[T]"]], Tuple[List["Tree[T]"], List["Tree[T]"]]]
        ] = None,
        post_process_fn: Optional[Callable[[List[T]], List[T]]] = None,
    ) -> List[T]:
        if partition_fn:
            lhs, rhs = partition_fn(self)
        else:
            lhs, rhs = self, []

        lhs_seq = [l for t in lhs for l in t.traversal(order, partition_fn, post_process_fn)]
        rhs_seq = [l for t in rhs for l in t.traversal(order, partition_fn, post_process_fn)]

        if post_process_fn:
            lhs_seq = post_process_fn(lhs_seq)
            rhs_seq = post_process_fn(rhs_seq)

        if order == TraversalOrder.PRE:
            return [self.label] + lhs_seq + rhs_seq
        elif order == TraversalOrder.IN:
            return lhs_seq + [self.label] + rhs_seq
        elif order == TraversalOrder.POST:
            return lhs_seq + rhs_seq + [self.label]
        else:
            raise ValueError(f"Unsupported traversal order {order}")

    def fold(self, f: Callable[[T, List[U]], U]) -> U:
        """Tree folding"""
        childs = [child.fold(f) for child in self]
        return f(self.label, childs)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            equal = self.label == other.label
            equal = equal and len(self) == len(other)
            for c1, c2 in zip(self, other):
                equal = equal and c1 == c2
            return equal
        return False

    def __repr__(self) -> str:
        childs = ", ".join(repr(child) for child in self)
        return f"{type(self).__name__}({repr(self.label)}, [{childs}])"
