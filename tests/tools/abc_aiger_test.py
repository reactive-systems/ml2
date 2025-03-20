"""ABC AIGER test"""

import pytest

from ml2.aiger import AIGERCircuit
from ml2.tools.abc_aiger import ABCAiger

TIMEOUT = 10


@pytest.mark.docker
def test_abc_aiger_functionality():
    abc_aiger = ABCAiger()
    assert abc_aiger.assert_identities
    assert abc_aiger.server_version is not None
    assert abc_aiger.functionality == []
    del abc_aiger


def test_aiger_convert_to_bin():
    abc_aiger = ABCAiger()
    circuit = AIGERCircuit.from_str(
        "aag 8 5 1 5 2\n2\n4\n6\n8\n10\n12 7\n0\n0\n1\n0\n16\n14 9 6\n16 14 12\ni0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4"
    )
    response = abc_aiger.convert_aasci_to_binary(circuit=circuit, parameters={"timeout": TIMEOUT})
    assert (
        response[0]
        == "aig 8 5 1 5 2\n7\n0\n0\n1\n0\n16\n\x05\x03\x02\x02i0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4\n"
    )
    del abc_aiger


def test_aiger_convert_from_bin():
    abc_aiger = ABCAiger()
    circuit = "aig 8 5 1 5 2\n7\n0\n0\n1\n0\n16\n\x05\x03\x02\x02i0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4\n"
    response = abc_aiger.convert_binary_to_aasci(
        circuit_bin=circuit, parameters={"timeout": TIMEOUT}
    )
    circuit = AIGERCircuit.from_str(
        "aag 8 5 1 5 2\n2\n4\n6\n8\n10\n12 7\n0\n0\n1\n0\n16\n14 9 6\n16 14 12\ni0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4"
    )
    assert circuit == response[0]
    del abc_aiger


def test_aiger_dot():
    abc_aiger = ABCAiger()
    circuit = AIGERCircuit.from_str(
        "aag 8 5 1 5 2\n2\n4\n6\n8\n10\n12 7\n0\n0\n1\n0\n16\n14 9 6\n16 14 12\ni0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4"
    )
    response = abc_aiger.convert_aiger_to_dot(circuit=circuit, parameters={"timeout": TIMEOUT})
    assert response[0].endswith(
        '{\n"2"[shape=box];\nI0[shape=triangle,color=blue];\nI0[label="i0"];\n"2"->I0[arrowhead=none];\n"4"[shape=box];\nI1[shape=triangle,color=blue];\nI1[label="i1"];\n"4"->I1[arrowhead=none];\n"6"[shape=box];\nI2[shape=triangle,color=blue];\nI2[label="i2"];\n"6"->I2[arrowhead=none];\n"8"[shape=box];\nI3[shape=triangle,color=blue];\nI3[label="i3"];\n"8"->I3[arrowhead=none];\n"10"[shape=box];\nI4[shape=triangle,color=blue];\nI4[label="i4"];\n"10"->I4[arrowhead=none];\n"14"->"8"[arrowhead=dot];\n"14"->"6"[arrowhead=none];\n"16"->"14"[arrowhead=none];\n"16"->"12"[arrowhead=none];\nO0[shape=triangle,color=blue];\nO0[label="o0"];\nO0 -> "0"[arrowhead=none];\nO1[shape=triangle,color=blue];\nO1[label="o1"];\nO1 -> "0"[arrowhead=none];\nO2[shape=triangle,color=blue];\nO2[label="o2"];\nO2 -> "0"[arrowhead=dot];\nO3[shape=triangle,color=blue];\nO3[label="o3"];\nO3 -> "0"[arrowhead=none];\nO4[shape=triangle,color=blue];\nO4[label="o4"];\nO4 -> "16"[arrowhead=none];\n"12"[shape=box,color=magenta];\nL0 [shape=diamond,color=magenta];\nL0[label="l0"];\nL0 -> "6"[arrowhead=dot];\nL0 -> "12"[style=dashed,color=magenta,arrowhead=none];\n"0"[color=red,shape=box];\n}\n'
    )

    del abc_aiger


def test_abc_simplify():
    abc_aiger = ABCAiger()
    circuit = AIGERCircuit.from_str(
        "aag 5 2 0 1 3\n2\n4\n10\n6 5 3\n8 4 2\n10 8 6\ni0 i0\ni1 i1\no0 o0"
    )
    simplified_circuit = AIGERCircuit.from_str("aag 2 2 0 1 0\n2\n4\n0\ni0 i0\ni1 i1\no0 o0")
    response = abc_aiger.aiger_simplify(circuit=circuit, parameters={"timeout": TIMEOUT})
    assert response[0][0] == circuit
    assert response[0][-1] == simplified_circuit
    del abc_aiger


def test_png():
    abc_aiger = ABCAiger()
    circuit = AIGERCircuit.from_str(
        "aag 8 5 1 5 2\n2\n4\n6\n8\n10\n12 7\n0\n0\n1\n0\n16\n14 9 6\n16 14 12\ni0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4"
    )
    response = abc_aiger.convert_aiger_to_dot(circuit=circuit, parameters={"timeout": TIMEOUT})
    response = abc_aiger.convert_dot_to_png(dot=response[0])
    del abc_aiger


def test_svg():
    abc_aiger = ABCAiger()
    circuit = AIGERCircuit.from_str(
        "aag 8 5 1 5 2\n2\n4\n6\n8\n10\n12 7\n0\n0\n1\n0\n16\n14 9 6\n16 14 12\ni0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4"
    )
    response = abc_aiger.convert_aiger_to_dot(circuit=circuit, parameters={"timeout": TIMEOUT})
    response = abc_aiger.convert_dot_to_svg(dot=response[0])
    assert response[0].startswith("<?xml") and "<svg" in response[0]
    del abc_aiger
