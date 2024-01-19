"""Mealy machine to sequence tokenizer"""

from typing import List

from grpc._channel import _InactiveRpcError

from ..prop import PropFormula
from ..tokenizers import ToSeqTokenizer
from ..tokenizers.expr_tokenizers.expr_to_seq_tokenizer import ExprToSeqTokenizer
from .mealy_machine import Condition, HoaHeader, MealyMachine, Transition

NEWLINE_TOKEN = "<n>"
STATE_TOKEN = "State: "
INIT_TOKEN = "Start: "
NUM_STATE_TOKEN = "States: "
PAR_TOKEN_OPEN = "["
PAR_TOKEN_CLOSE = "] "
BODY_START_TOKEN = "--BODY--"
BODY_END_TOKEN = "--END--"
EDGE_DELIMITER_TOKEN = "<e>"
COND_DELIMITER_TOKEN = "<c>"


class MealyToSeqTokenizer(ToSeqTokenizer[MealyMachine]):
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        method: str = "transitions",
        include_body_tokens: bool = None,
        notation: str = None,
        **kwargs,
    ):
        self.method = method
        self.inputs = inputs
        self.outputs = outputs
        if method == "transitions":
            from ml2.tools.spot import Spot

            if notation is None:
                notation = "prefix"
            self.notation = notation
            self.include_body_tokens = None
            self.spot = Spot()
            self.expr_tokenizer = ExprToSeqTokenizer(self.notation, pad=128, dtype=PropFormula)
        elif method == "simplified_hoa":
            if include_body_tokens is None:
                include_body_tokens = False
            if notation is None:
                notation = "infix"
            self.notation = notation
            self.include_body_tokens = include_body_tokens
        else:
            raise ValueError

        super().__init__(dtype=MealyMachine, **kwargs)

    @staticmethod
    def token_error(token_is, token_should) -> None:
        if token_is != token_should:
            raise Exception(
                "Token Decode Error: Token {} expected but token {} found".format(
                    token_is, token_should
                )
            )

    def encode_tokens_transitions(self, data: MealyMachine) -> List[str]:
        try:
            transitions = self.spot.extractTransitions(
                data.to_hoa(realizable=True)
            )  # ugly hack but realizability does not matter here
        except _InactiveRpcError:
            # raise Exception(
            #     "Token Encode Error (using spot -- probably too big). File: {}".format(
            #         data.to_hoa(realizable=True)
            #     )
            # )
            print("_InactiveRpcError")
            return [EDGE_DELIMITER_TOKEN]  # ugly hack
        tokens = []
        for edge in [
            Transition(t["src"], t["dst"], Condition(formula=t["cond"], notation=self.notation))
            for t in transitions
        ]:
            tokens.append(str(edge.src))
            tokens.append(COND_DELIMITER_TOKEN)
            tokens = tokens + self.expr_tokenizer.encode_tokens(edge.cond)
            tokens.append(COND_DELIMITER_TOKEN)
            tokens.append(str(edge.dst))
            tokens.append(EDGE_DELIMITER_TOKEN)
        return tokens

    def decode_tokens_transitions(self, tokens: List[str]) -> MealyMachine:
        transitions = []
        while tokens:
            src = int(tokens.pop(0))
            MealyToSeqTokenizer.token_error(tokens.pop(0), COND_DELIMITER_TOKEN)
            cond = ""
            while tokens[0] != COND_DELIMITER_TOKEN:
                cond = cond + " " + tokens.pop(0)
            MealyToSeqTokenizer.token_error(tokens.pop(0), COND_DELIMITER_TOKEN)
            dst = int(tokens.pop(0))
            MealyToSeqTokenizer.token_error(tokens.pop(0), EDGE_DELIMITER_TOKEN)
            transitions.append(
                Transition(src, dst, Condition(formula=cond, notation=self.notation))
            )
        header = HoaHeader.mealy_header(
            num_states=len(set([i.src for i in transitions] + [i.dst for i in transitions])),
            inputs=self.inputs,
            outputs=self.outputs,
            init_state=0,  # bfs or dfs ensures that
        )
        return self.dtype(header=header, transitions=transitions)

    def encode_tokens_hoa(self, data: MealyMachine) -> List[str]:
        tokens = []
        tokens.append(NUM_STATE_TOKEN)
        tokens.append(str(data.num_states))
        tokens.append(NEWLINE_TOKEN)
        tokens.append(INIT_TOKEN)
        tokens.append(str(data.header.init_state))
        tokens.append(NEWLINE_TOKEN)
        if self.include_body_tokens:
            tokens.append(BODY_START_TOKEN)
            tokens.append(NEWLINE_TOKEN)
        for i in range(data.num_states):
            tokens.append(STATE_TOKEN)
            tokens.append(str(i))
            tokens.append(NEWLINE_TOKEN)
            tokens = tokens + [
                y
                for x in [
                    [PAR_TOKEN_OPEN]
                    + [char for char in t.cond.to_hoa_str(data.header.aps, self.notation)]
                    + [PAR_TOKEN_CLOSE]
                    + [str(t.dst)]
                    + [NEWLINE_TOKEN]
                    for t in (filter(lambda t: t.src == i, data.transitions))
                ]
                for y in x
            ]
        if self.include_body_tokens:
            tokens.append(BODY_END_TOKEN)
        return tokens

    def decode_tokens_hoa(self, tokens: List[str]) -> MealyMachine:
        self.token_error(tokens.pop(0), NUM_STATE_TOKEN)
        num_states = int(tokens.pop(0))
        self.token_error(tokens.pop(0), NEWLINE_TOKEN)
        self.token_error(tokens.pop(0), INIT_TOKEN)
        init_state = int(tokens.pop(0))
        self.token_error(tokens.pop(0), NEWLINE_TOKEN)
        header = HoaHeader.mealy_header(
            num_states=num_states, init_state=init_state, inputs=self.inputs, outputs=self.outputs
        )
        if self.include_body_tokens:
            self.token_error(tokens.pop(0), BODY_START_TOKEN)
            self.token_error(tokens.pop(0), NEWLINE_TOKEN)
            self.token_error(tokens.pop(len(tokens) - 1), BODY_END_TOKEN)
        transitions = []
        while tokens:
            self.token_error(tokens.pop(0), STATE_TOKEN)
            src = int(tokens.pop(0))
            self.token_error(tokens.pop(0), NEWLINE_TOKEN)
            while tokens and tokens[0] == PAR_TOKEN_OPEN:
                tokens.pop(0)
                cond = ""
                while tokens[0] != PAR_TOKEN_CLOSE:
                    cond = cond + " " + tokens.pop(0)
                self.token_error(tokens.pop(0), PAR_TOKEN_CLOSE)
                dst = int(tokens.pop(0))
                self.token_error(tokens.pop(0), NEWLINE_TOKEN)
                transitions.append(
                    Transition(src, dst, Condition.from_hoa_str(cond, header.aps, self.notation))
                )
        return self.dtype(header=header, transitions=transitions)

    def encode_tokens(self, data: MealyMachine, **kwargs) -> List[str]:
        if self.method == "simplified_hoa":
            return self.encode_tokens_hoa(data=data)
        elif self.method == "transitions":
            return self.encode_tokens_transitions(data=data)
        else:
            raise ValueError

    def decode_tokens(self, tokens: List[str], **kwargs) -> MealyMachine:
        if self.method == "simplified_hoa":
            return self.decode_tokens_hoa(tokens=tokens)
        elif self.method == "transitions":
            return self.decode_tokens_transitions(tokens=tokens)
        else:
            raise ValueError
