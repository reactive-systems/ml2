"""Inspired by the nltk tree class https://www.nltk.org/_modules/nltk/tree.html"""


class Tree(list):
    def __init__(self, label, children: list):

        list.__init__(self, children)
        self.label = label

    def leaves(self):
        """Returns a list of (labels of) leaves"""
        if not self:
            return [self.label]
        leaves = []
        for child in self:
            leaves.extend(child.leaves())
        return leaves

    def size(self):
        """Returns the size of the tree, i.e., the number of nodes"""
        size = 1
        for child in self:
            size += child.size()
        return size

    def breadth(self):
        """Returns the breadth of the tree, i.e., the number of leaves"""
        return len(self.leaves())

    def degree(self):
        """Returns the degree of the tree, i.e., the maximum number of children a node has"""
        degree = len(self)
        for child in self:
            degree = max(degree, child.degree())
        return degree

    def height(self):
        """Returns the height of the tree, i.e., the length of the longest path from the root node to a leaf node"""
        max_child_height = -1
        for child in self:
            max_child_height = max(max_child_height, child.height())
        return max_child_height + 1

    def to_str(self, join_fn):
        """Returns a string respresenting the tree given a function that combines the label with the string respresentation of the childs"""
        childs = [child.to_str(join_fn) for child in self]
        return join_fn(self.label, childs)

    def __repr__(self):
        childs = ", ".join(repr(child) for child in self)
        return f"{type(self).__name__}({repr(self.label)}, [{childs}])"
