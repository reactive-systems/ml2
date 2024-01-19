"""LTL synthesis solution tokenizer config test"""

from ml2.aiger import AIGERCircuit
from ml2.ltl.ltl_syn import LTLSynSolution, LTLSynSolutionToSeqTokenizer, LTLSynStatus

LTL_SYN_SOLUTION_TO_SEQ_TOKENIZER_CONFIG = {
    "name": "ltl-syn-solution-to-seq-tokenizer",
    "project": "test",
    "components": ["header", "inputs", "latches", "outputs", "ands"],
    "inputs": ["i0", "i1", "i2", "i3", "i4"],
    "outputs": ["o0", "o1", "o2", "o3", "o4"],
    "eos": True,
    "start": False,
    "pad": 48,
    "vocabulary": {
        "name": "ltl-syn-solution-to-seq-tokenizer/vocabulary",
        "project": "test",
        "token_to_id": {
            "<p>": 0,
            "0": 1,
            "1": 2,
            "2": 3,
            "3": 4,
            "4": 5,
            "5": 6,
            "6": 7,
            "7": 8,
            "8": 9,
            "9": 10,
            "10": 11,
            "11": 12,
            "12": 13,
            "13": 14,
            "14": 15,
            "15": 16,
            "16": 17,
            "17": 18,
            "18": 19,
            "<n>": 20,
            "aag": 21,
            "realizable": 22,
            "unrealizable": 23,
            "<e>": 24,
        },
    },
}


def test_ltl_syn_solution_tokenizer_config():
    tokenizer = LTLSynSolutionToSeqTokenizer.from_config(LTL_SYN_SOLUTION_TO_SEQ_TOKENIZER_CONFIG)
    status = LTLSynStatus("realizable")
    circuit = circuit = AIGERCircuit.from_str(
        "aag 9 5 1 5 3\n2\n4\n6\n8\n10\n12 18\n1\n1\n1\n0\n16\n14 13 5\n16 15 6\n18 15 7\ni0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4"
    )
    ltl_syn_solution = LTLSynSolution(status=status, circuit=circuit)
    encoding = tokenizer.encode(ltl_syn_solution)
    assert encoding.ids == [
        22,
        10,
        6,
        2,
        6,
        4,
        20,
        3,
        20,
        5,
        20,
        7,
        20,
        9,
        20,
        11,
        20,
        13,
        19,
        20,
        2,
        20,
        2,
        20,
        2,
        20,
        1,
        20,
        17,
        20,
        15,
        14,
        6,
        20,
        17,
        16,
        7,
        20,
        19,
        16,
        8,
        24,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    config = tokenizer.get_config()
    assert config["name"] == "ltl-syn-solution-to-seq-tokenizer"
    assert config["project"] == "test"
    assert config["dtype"] == "LTLSynSolution"
    assert config["components"] == ["header", "inputs", "latches", "outputs", "ands"]
    assert config["inputs"] == ["i0", "i1", "i2", "i3", "i4"]
    assert config["outputs"] == ["o0", "o1", "o2", "o3", "o4"]
    assert config["eos"]
    assert not config["start"]
    assert config["pad"] == 48
    assert config["type"] == "LTLSynSolutionToSeqTokenizer"
