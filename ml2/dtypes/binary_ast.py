"""Abstract syntax tree class"""

import logging
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

from .seq import Seq
from .tree import TraversalOrder, Tree

T = TypeVar("T")


class TPEFormat(Enum):
    BRANCHUP = "branch-up"
    BRANCHDOWN = "branch-down"


T = TypeVar("T")


class BinaryAST(Tree[str], Seq):
    def __init__(self, label: str, lhs: "BinaryAST" = None, rhs: "BinaryAST" = None):
        children = []
        if lhs is not None:
            children.append(lhs)
        if rhs is not None:
            children.append(rhs)
        super().__init__(label=label, children=children)

    @property
    def has_lhs(self) -> bool:
        return len(self) > 0

    @property
    def has_rhs(self) -> bool:
        return len(self) > 1

    @property
    def is_binary(self) -> bool:
        return self.has_rhs

    @property
    def is_unary(self) -> bool:
        return not self.has_rhs and self.has_lhs

    @property
    def is_leaf(self) -> bool:
        return not self.has_lhs

    @property
    def lhs(self) -> "BinaryAST":
        if len(self) < 1:
            raise ValueError()
        return self[0]

    @property
    def rhs(self) -> "BinaryAST":
        if len(self) < 2:
            raise ValueError()
        return self[1]

    def size(self, *, notation: str = None, **kwargs) -> int:
        if notation is None:
            return super().size(**kwargs)
        else:
            return len(self.to_tokens(notation=notation, **kwargs))

    @classmethod
    def big(
        cls,
        operator: str,
        idx_iterable: Iterable[T],
        body: Callable[[T], "BinaryAST"],
    ) -> "BinaryAST":
        """The big operator (e.g. big_and or big_or). The implemented version of the mathematical operator(s) bigwedge_{i=1}^9 a_i lor b

        Args:
            operator (str): Operator which concatenates the instances.
            idx_iterable (Iterable[T]): Collection of indexes which span the instances (typically set or list of ints).
            body (Callable[[T], BinaryAST]])): A unction creating the body of the operator which gets instantiated. The function is parameterized by the index from idx_iterable.

        Returns:
            BinaryAST: The new BinaryAST.
        """
        new_asts = []
        for idx in idx_iterable:
            new_asts.append(body(idx))
        new_ast = new_asts.pop()
        while len(new_asts):
            new_ast = BinaryAST(label=operator, lhs=new_asts.pop(), rhs=new_ast)
        return new_ast

    def to_tokens(
        self,
        *,
        notation: str = "infix",
        precedence: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[str]:
        # example for precedence in LTL:
        # all missing operators or missing associativity will lead to unambiguous parenthesizing:
        # precedence = [
        #     # low
        #     {"operator": ["<->", "->"]},
        #     {"assoc": "left", "operator": ["^"]},
        #     {"assoc": "left", "operator": ["|"]},
        #     {"assoc": "left", "operator": ["&"]},
        #     {"operator": ["U", "W", "R"]},
        #     {"assoc": "right", "operator": ["X", "!", "F", "G"]},
        #     # high
        # ]
        if precedence is None:
            precedence = []

        assoc_map = {op: p["assoc"] for p in precedence if "assoc" in p for op in p["operator"]}
        acc = 0, None
        p = (acc := (acc[0] + 1, p["operator"]) for p in precedence)
        precedence_map = {op: e[0] for e in p for op in e[1]}

        def add_pars(tokens: List[str], tree: "BinaryAST", child: str) -> List[str]:
            if tokens:
                child_node = tree.lhs if child == "left" or tree.is_unary else tree.rhs
                if (
                    not child_node.is_leaf or precedence == []
                ):  # not if its a leaf except for fully parenthesized
                    if tree.label not in precedence_map or child_node.label not in precedence_map:
                        tokens = ["("] + tokens + [")"]
                    elif precedence_map[child_node.label] < precedence_map[tree.label]:
                        tokens = ["("] + tokens + [")"]
                    elif precedence_map[child_node.label] == precedence_map[tree.label]:
                        if tree.label not in assoc_map or assoc_map[tree.label] != child:
                            tokens = ["("] + tokens + [")"]
            return tokens

        def binary_part(childs: List[Tree[T]]) -> Tuple[List[Tree[T]], List[Tree[T]]]:
            if len(childs) == 0:
                return [], []
            elif len(childs) == 1:
                return [], childs
            elif len(childs) == 2:
                return [childs[0]], [childs[1]]
            else:
                raise ValueError("Binary tree with more than two childs")

        if notation == "infix":
            return self.traversal(TraversalOrder.IN, binary_part, add_pars)
        elif notation == "infix-no-pars":
            return self.traversal(TraversalOrder.IN, binary_part)
        elif notation == "prefix":
            return self.traversal(TraversalOrder.PRE, binary_part)
        else:
            raise ValueError(f"Unsupported expression notation {notation}")

    def to_str(self, *, space: bool = True, **kwargs) -> str:
        token_list = self.to_tokens(**kwargs)
        if space:
            return " ".join(token_list)
        else:
            return "".join(token_list)

    def tree_positional_encoding(self, notation: str, format: TPEFormat) -> List[List[int]]:
        """Returns a list of binary lists where each list represents the position of a node in the abstract syntax tree as a sequence of steps along tree branches starting in the root node with each step going left ([1,0]) or going right([0,1]). The ordering of the lists is given by the notation. format specifies whether branching choices are added at the first position or at the last position of a list.
        Args:
            notation: str
            format: ml2.data.ast.TPEFormat
        Example: Given the abstract syntax tree representing the LTL formula aUb&Xc. Depending on the parameters the following lists are returned.
            notation=INFIXNOPARS, format=BRANCHUP: [[1,0,1,0],[1,0],[0,1,1,0],[],[0,1],[1,0,0,1]]
            notation=INFIXNOPARS, format=BRANCHDOWN: [[1,0,1,0],[1,0],[1,0,0,1],[],[0,1],[0,1,1,0]]
            notation=PREFIX, format=BRANCHUP: [[], [1,0], [1,0,1,0], [0,1,1,0], [0,1], [1,0,0,1]]
            notation=PREFIX, format=BRANCHDOWN: [[],[1,0],[1,0,1,0],[1,0,0,1],[0,1],[0,1,1,0]]
        """
        new_enc = [[]] if self.label != "" else []
        if not (self.has_lhs or self.has_rhs):
            return new_enc
        elif not self.has_lhs:
            rhs_pos_list = self.rhs.tree_positional_encoding(notation, format)
            if format == TPEFormat.BRANCHUP:
                rhs_pos_list = [l + [0, 1] for l in rhs_pos_list]
            else:
                rhs_pos_list = [[0, 1] + l for l in rhs_pos_list]
            return new_enc + rhs_pos_list
        elif not self.has_rhs:
            lhs_pos_list = self.lhs.tree_positional_encoding(notation, format)
            if format == TPEFormat.BRANCHUP:
                lhs_pos_list = [l + [1, 0] for l in lhs_pos_list]
            else:
                lhs_pos_list = [[1, 0] + l for l in lhs_pos_list]
            return new_enc + lhs_pos_list
        else:
            lhs_pos_list = self.lhs.tree_positional_encoding(notation, format)
            rhs_pos_list = self.rhs.tree_positional_encoding(notation, format)
            if format == TPEFormat.BRANCHUP:
                # due to the recursion the branching choice is added at the end of the list if add_first is true
                lhs_pos_list = [l + [1, 0] for l in lhs_pos_list]
                rhs_pos_list = [l + [0, 1] for l in rhs_pos_list]
            else:
                lhs_pos_list = [[1, 0] + l for l in lhs_pos_list]
                rhs_pos_list = [[0, 1] + l for l in rhs_pos_list]
            if notation == "prefix":
                return new_enc + lhs_pos_list + rhs_pos_list
            if notation == "infix-no-pars":
                return lhs_pos_list + new_enc + rhs_pos_list
            else:
                logging.critical(f"Unsupported notation {notation}")

    def commutative_equal(self, other, commutative_operators: List[str]) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.label != other.label:
            return False
        if len(self) != len(other):
            return False
        child_equal = True
        for c1, c2 in zip(self, other):
            child_equal = child_equal and c1.commutative_equal(c2, commutative_operators)
        if self.label in commutative_operators:
            child_equal = child_equal or (
                self.lhs.commutative_equal(other.rhs, commutative_operators)
                and self.rhs.commutative_equal(other.lhs, commutative_operators)
            )
        return child_equal
