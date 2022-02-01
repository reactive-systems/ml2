from itertools import combinations


class Pattern:
    def fill(self, inputs: list, outputs: list) -> str:
        raise NotImplementedError()


class UnaryPattern(Pattern):
    def __init__(self, child):
        self.child = child
        self.num_inputs = child.num_inputs
        self.num_outputs = child.num_outputs


class BinaryPattern(Pattern):
    def __init__(self, child1, child2):
        self.child1 = child1
        self.child2 = child2
        self.num_inputs = child1.num_inputs + child2.num_inputs
        self.num_outputs = child1.num_outputs + child2.num_outputs


class KAryPattern(Pattern):
    def __init__(self, childs):
        self.childs = childs
        self.num_inputs = sum([child.num_inputs for child in childs])
        self.num_outputs = sum([child.num_outputs for child in childs])

    def filled_childs(self, inputs, outputs):
        input_index = 0
        output_index = 0
        filled_childs = []
        for child in self.childs:
            filled_childs.append(child.fill(inputs[input_index:], outputs[output_index:]))
            input_index += child.num_inputs
            output_index += child.num_outputs
        return filled_childs


class Response(BinaryPattern):
    def __init__(self, child1, child2):
        assert child1.num_outputs == 0
        assert child2.num_inputs == 0
        super().__init__(child1, child2)

    def fill(self, inputs, outputs):
        return f"G ( ( {self.child1.fill(inputs, outputs)} ) -> ( {self.child2.fill(inputs, outputs)} ) )"


class Precedence(BinaryPattern):
    def __init__(self, child1, child2):
        assert child1.num_inputs == 0
        assert child2.num_outputs == 0
        super().__init__(child1, child2)

    def fill(self, inputs, outputs):
        return f"G ( ( {self.child1.fill(inputs, outputs)} ) -> ( {self.child2.fill(inputs, outputs)} ) )"


class Correspondece(BinaryPattern):
    def fill(self, inputs, outputs):
        return f"G ( ( {self.child1.fill(inputs, outputs)} ) <-> ( {self.child2.fill(inputs, outputs)} ) )"


class AfterNSteps(UnaryPattern):
    def __init__(self, child, n):
        self.n = n
        super().__init__(child)

    def fill(self, inputs, outputs):
        filled_child = self.child.fill(inputs, outputs)
        if len(filled_child) > 2:
            filled_child = f"( {filled_child} )"
        return f'{"X " * self.n}{filled_child}'


class ForNSteps(UnaryPattern):
    def __init__(self, child, n):
        self.n = n
        super().__init__(child)

    def fill(self, inputs, outputs):
        filled_child = self.child.fill(inputs, outputs)
        if self.n > 1:
            if len(filled_child) > 2:
                filled_child = f"( {filled_child} )"
            return " & ".join([f'{"X " * i}{filled_child}' for i in range(self.n)])
        return filled_child


class WithinNSteps(UnaryPattern):
    def __init__(self, child, n):
        self.n = n
        super().__init__(child)

    def fill(self, inputs, outputs):
        filled_child = self.child.fill(inputs, outputs)
        if self.n > 1:
            if len(filled_child) > 2:
                filled_child = f"( {filled_child} )"
            return " | ".join([f'{"X " * i}{filled_child}' for i in range(self.n)])
        return filled_child


class Eventually(KAryPattern):
    def fill(self, inputs, outputs):
        input_index = 0
        output_index = 0
        filled_childs = []
        for child in self.childs:
            filled_childs.append(child.fill(inputs[input_index:], outputs[output_index:]))
            input_index += child.num_inputs
            output_index += child.num_outputs
        return " & ".join([f"F ( {filled_child} )" for filled_child in filled_childs])


class Globally(KAryPattern):
    def fill(self, inputs, outputs):
        input_index = 0
        output_index = 0
        filled_childs = []
        for child in self.childs:
            filled_childs.append(child.fill(inputs[input_index:], outputs[output_index:]))
            input_index += child.num_inputs
            output_index += child.num_outputs
        return " | ".join([f"G ( {filled_child} )" for filled_child in filled_childs])


class Recurrence(Eventually):
    def fill(self, inputs, outputs):
        return f"G ( {super().fill(inputs, outputs)} )"


class Persistence(Globally):
    def fill(self, inputs, outputs):
        return f"F ( {super().fill(inputs, outputs)} )"


class CNFMutualExclusion(KAryPattern):
    def fill(self, inputs, outputs):
        filled_childs = [
            f"! {filled_child}" for filled_child in self.filled_childs(inputs, outputs)
        ]
        return " & ".join(
            [
                f"( {filled_childs[i]} | {filled_childs[j]} )"
                for (i, j) in combinations(range(len(self.childs)), 2)
            ]
        )


class DNFMutualExclusion(KAryPattern):
    def fill(self, inputs, outputs):
        filled_childs = [
            f"! {filled_child}" for filled_child in self.filled_childs(inputs, outputs)
        ]
        return " | ".join(
            [
                " & ".join(filled_childs[:i] + filled_childs[i + 1 :])
                for i in range(len(self.childs))
            ]
        )


class FactorizedCNFMutualExclusion(KAryPattern):
    def fill(self, inputs, outputs):
        filled_childs = [
            f"! {filled_child}" for filled_child in self.filled_childs(inputs, outputs)
        ]
        if len(self.childs) != 3:
            # TODO implement general case
            raise NotImplementedError()
        return f"({filled_childs[0]} | ({filled_childs[1]} & {filled_childs[2]})) & ({filled_childs[1]} | {filled_childs[2]})"


class FactorizedDNFMutualExclusion(KAryPattern):
    def fill(self, inputs, outputs):
        filled_childs = [
            f"! {filled_child}" for filled_child in self.filled_childs(inputs, outputs)
        ]
        if len(self.childs) != 3:
            # TODO implement general case
            raise NotImplementedError()
        return f"({filled_childs[0]} & ({filled_childs[1]} | {filled_childs[2]})) | ({filled_childs[1]} & {filled_childs[2]})"


class AndInput(Pattern):
    def __init__(self):
        self.num_inputs = 2
        self.num_outputs = 0

    def fill(self, inputs, outputs):
        return f"{inputs[0]} & {inputs[1]}"


class OrInput(Pattern):
    def __init__(self):
        self.num_inputs = 2
        self.num_outputs = 0

    def fill(self, inputs, outputs):
        return f"{inputs[0]} | {inputs[1]}"


class AtomicInput(Pattern):
    def __init__(self):
        self.num_inputs = 1
        self.num_outputs = 0

    def fill(self, inputs, outputs):
        return inputs[0]


class AndOutput(Pattern):
    def __init__(self):
        self.num_inputs = 0
        self.num_outputs = 2

    def fill(self, inputs, outputs):
        return f"{outputs[0]} & {outputs[1]}"


class OrOutput(Pattern):
    def __init__(self):
        self.num_inputs = 0
        self.num_outputs = 2

    def fill(self, inputs, outputs):
        return f"{outputs[0]} | {outputs[1]}"


class AtomicOutput(Pattern):
    def __init__(self):
        self.num_inputs = 0
        self.num_outputs = 1

    def fill(self, inputs, outputs):
        return outputs[0]
