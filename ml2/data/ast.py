"""Abstract syntax tree classes"""

from enum import Enum
import logging

from .expr import ExprNotation
from .tree import Tree


class TPEFormat(Enum):
    BRANCHUP = "branch-up"
    BRANCHDOWN = "branch-down"


class BinaryAST(Tree):
    def __init__(self, label, lhs=None, rhs=None):
        self.lhs = lhs
        self.rhs = rhs
        children = []
        if self.lhs is not None:
            children.append(self.lhs)
        if self.rhs is not None:
            children.append(self.rhs)
        super().__init__(label, children)

    def to_list(self, notation: ExprNotation) -> list:
        # TODO remove unnecessary paranthesis
        """
        Args:
            notation: ml2.data.expr.ExprNotation
        """
        result = [self.label]
        if self.lhs is None and self.rhs is None:
            pass
        elif self.lhs is None:
            if notation == ExprNotation.INFIX:
                result.extend(["("] + self.rhs.to_list(notation) + [")"])
            else:
                result.extend(self.rhs.to_list(notation))
        elif self.rhs is None:
            if notation == ExprNotation.INFIX:
                result.extend(["("] + self.lhs.to_list(notation) + [")"])
            else:
                result.extend(self.lhs.to_list(notation))
        else:
            if notation == ExprNotation.INFIX:
                result = (
                    ["("]
                    + self.lhs.to_list(notation)
                    + [")"]
                    + result
                    + ["("]
                    + self.rhs.to_list(notation)
                    + [")"]
                )
            elif notation == ExprNotation.INFIXNOPARS:
                result = self.lhs.to_list(notation) + result + self.rhs.to_list(notation)
            elif notation == ExprNotation.PREFIX:
                result += self.lhs.to_list(notation) + self.rhs.to_list(notation)
        return result

    def to_str(self, notation: ExprNotation, space: bool = True) -> str:
        """
        Args:
            notation: ml2.data.expr.ExprNotation
        """
        token_list = self.to_list(notation)
        if space:
            return " ".join(token_list)
        else:
            return "".join(token_list)

    def tree_positional_encoding(self, notation: ExprNotation, format: TPEFormat):
        # TODO support for infix notation, combine with to_list?
        """Returns a list of binary lists where each list represents the position of a node in the abstract syntax tree as a sequence of steps along tree branches starting in the root node with each step going left ([1,0]) or going right([0,1]). The ordering of the lists is given by the notation. format specifies whether branching choices are added at the first position or at the last position of a list.
        Args:
            notation: ml2.data.expr.ExprNotation
            format: ml2.data.ast.TPEFormat
        Example: Given the abstract syntax tree representing the LTL formula aUb&Xc. Depending on the parameters the following lists are returned.
            notation=INFIXNOPARS, format=BRANCHUP: [[1,0,1,0],[1,0],[0,1,1,0],[],[0,1],[1,0,0,1]]
            notation=INFIXNOPARS, format=BRANCHDOWN: [[1,0,1,0],[1,0],[1,0,0,1],[],[0,1],[0,1,1,0]]
            notation=PREFIX, format=BRANCHUP: [[], [1,0], [1,0,1,0], [0,1,1,0], [0,1], [1,0,0,1]]
            notation=PREFIX, format=BRANCHDOWN: [[],[1,0],[1,0,1,0],[1,0,0,1],[0,1],[0,1,1,0]]
        """
        if self.lhs is None and self.rhs is None:
            return [[]]
        elif self.lhs is None:
            rhs_pos_list = self.rhs.tree_positional_encoding(notation, format)
            if format == TPEFormat.BRANCHUP:
                rhs_pos_list = [l + [0, 1] for l in rhs_pos_list]
            else:
                rhs_pos_list = [[0, 1] + l for l in rhs_pos_list]
            return [[]] + rhs_pos_list
        elif self.rhs is None:
            lhs_pos_list = self.lhs.tree_positional_encoding(notation, format)
            if format == TPEFormat.BRANCHUP:
                lhs_pos_list = [l + [1, 0] for l in lhs_pos_list]
            else:
                lhs_pos_list = [[1, 0] + l for l in lhs_pos_list]
            return [[]] + lhs_pos_list
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
            if notation == ExprNotation.PREFIX:
                return [[]] + lhs_pos_list + rhs_pos_list
            if notation == ExprNotation.INFIXNOPARS:
                return lhs_pos_list + [[]] + rhs_pos_list
            else:
                logging.critical(f"Unsupported notation {notation}")
