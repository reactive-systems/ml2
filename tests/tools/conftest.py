import pytest

from ml2.aiger import AIGERCircuit
from ml2.ltl import DecompLTLSpec
from ml2.tools.ltl_tool import ToolLTLMCProblem

TIMEOUT = 10


@pytest.fixture()
def sat_real_ltl_mc_problem():
    spec = DecompLTLSpec.from_dict(
        {
            "guarantees": ["G (r1 -> F g1)", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    satisfying_system = AIGERCircuit.from_str(
        "aag 3 2 1 2 0\n2\n4\n6 7\n7\n6\ni0 r1\ni1 r2\nl0 l0\no0 g1\no1 g2"
    )
    return ToolLTLMCProblem(
        parameters={"timeout": TIMEOUT},
        realizable=True,
        specification=spec,
        circuit=satisfying_system,
    )


@pytest.fixture()
def sat_unreal_ltl_mc_problem():
    spec = DecompLTLSpec.from_dict(
        {
            "guarantees": ["G (r1 -> ((! g2) U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    satisfying_system = AIGERCircuit.from_str(
        "aag 2 2 0 2 0\n2\n4\n1\n1\ni0 g1\ni1 g2\no0 r1\no1 r2"
    )
    return ToolLTLMCProblem(
        parameters={"timeout": TIMEOUT},
        realizable=False,
        specification=spec,
        circuit=satisfying_system,
    )


@pytest.fixture()
def unsat_real_ltl_mc_problem():
    spec = DecompLTLSpec.from_dict(
        {
            "guarantees": ["G (r1 -> F g1)", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    violating_system = AIGERCircuit.from_str(
        "aag 3 2 1 2 0\n2\n4\n6 7\n7\n7\ni0 r1\ni1 r2\nl0 l0\no0 g1\no1 g2"
    )
    return ToolLTLMCProblem(
        parameters={"timeout": TIMEOUT},
        realizable=True,
        specification=spec,
        circuit=violating_system,
    )


@pytest.fixture()
def unsat_unreal_ltl_mc_problem():
    spec = DecompLTLSpec.from_dict(
        {
            "guarantees": ["G (r1 -> ((! g2) U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    satisfying_system = AIGERCircuit.from_str(
        "aag 2 2 0 2 0\n2\n4\n0\n1\ni0 g1\ni1 g2\no0 r1\no1 r2"
    )
    return ToolLTLMCProblem(
        parameters={"timeout": TIMEOUT},
        realizable=False,
        specification=spec,
        circuit=satisfying_system,
    )


@pytest.fixture()
def invalid_ltl_mc_problem():
    spec = DecompLTLSpec.from_dict(
        {
            "guarantees": ["G (r1 -> F g1)", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    satisfying_system = AIGERCircuit.from_str(
        "aag 3 2 1 2 0\n2\n4\n6 7\n7\n6\ni0 r1\ni1 r2\nl0 l0\no0 g1\no1 g2"
    )
    return ToolLTLMCProblem(
        parameters={"timeout": TIMEOUT},
        realizable=False,
        specification=spec,
        circuit=satisfying_system,
    )


@pytest.fixture()
def timeout_ltl_mc_problem():
    spec = DecompLTLSpec.from_dict(
        {
            "guarantees": ["G (r1 -> F g1)", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    satisfying_system = AIGERCircuit.from_str(
        "aag 3 2 1 2 0\n2\n4\n6 7\n7\n6\ni0 r1\ni1 r2\nl0 l0\no0 g1\no1 g2"
    )
    return ToolLTLMCProblem(
        parameters={"timeout": 0},
        realizable=True,
        specification=spec,
        circuit=satisfying_system,
    )
