"""LTL data"""


class LTLData:
    def __init__(self, formulas):
        self.dataset = formulas
        print(f"Successfully constructed dataset of {len(self.dataset)} LTL formulas")

    def filter(self, predicate):
        self.dataset = [formula for formula in self.dataset if predicate(formula)]
        print(f"Filtered dataset contains {len(self.dataset)} LTL formulas")

    def generator(self):
        for formula in self.dataset:
            yield formula

    def size(self):
        return len(self.dataset)

    def to_file(self, filepath):
        with open(filepath, "w") as file:
            for formula in self.dataset:
                file.write(formula.to_str())

    @classmethod
    def from_iterable(cls, formulas):
        """Constructs LTL data from an iterable of LTL formula strings"""
        return cls(formulas)

    @classmethod
    def from_file(cls, filepath):
        """Constructs LTL data from a text file containing LTL formula strings"""
        formulas = []
        with open(filepath, "r") as file:
            formula = file.readline().strip()
            formulas.append(formula)
        return cls(formulas)
