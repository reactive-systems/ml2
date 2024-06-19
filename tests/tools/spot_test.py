"""Spot test"""

import pytest

from ml2.aiger import AIGERCircuit
from ml2.ltl import DecompLTLSpec, LTLFormula
from ml2.ltl.ltl_mc import LTLMCStatus
from ml2.ltl.ltl_sat import LTLSatStatus
from ml2.mealy import MealyMachine
from ml2.tools.ltl_tool import ToolLTLMCProblem, ToolLTLSynProblem
from ml2.tools.spot import Spot
from ml2.trace import SymbolicTrace, TraceMCStatus

TIMEOUT = 10


@pytest.mark.docker
def test_spot_functionality():
    spot = Spot()
    assert spot.assert_identities
    assert spot.server_version is not None
    assert spot.functionality == [
        "FUNCTIONALITY_LTL_AIGER_MODELCHECKING",
        "FUNCTIONALITY_LTL_MEALY_MODELCHECKING",
        "FUNCTIONALITY_LTL_EQUIVALENCE",
        "FUNCTIONALITY_LTL_TRACE_MODELCHECKING",
        "FUNCTIONALITY_RANDLTL",
        "FUNCTIONALITY_AIGER_TO_MEALY",
        "FUNCTIONALITY_MEALY_TO_AIGER",
        "FUNCTIONALITY_LTL_AIGER_SYNTHESIS",
    ]
    del spot


@pytest.mark.docker
def test_spot_equiv():
    spot = Spot()
    f = LTLFormula.from_str("a U b")
    g = LTLFormula.from_str("b | (a & X (a U b))")
    assert spot.check_equiv(f, g).equiv
    del spot


@pytest.mark.docker
def test_spot_incl_1():
    spot = Spot()
    f = LTLFormula.from_str("a U b")
    g = LTLFormula.from_str("b | (a & X (a U b))")
    assert spot.inclusion(f, g).equiv
    del spot


@pytest.mark.docker
def test_spot_equiv_renaming():
    spot = Spot()
    f = LTLFormula.from_str("(F G ! x1) & F x1 & !x2 & x3 & (x4 | !x4) ")
    g = LTLFormula.from_str("!a & F (b & X G ! b) & c")
    assert spot.check_equiv_renaming(f, g).equiv
    del spot


@pytest.mark.docker
def test_spot_not_equiv():
    spot = Spot()
    f = LTLFormula.from_str("a U b")
    g = LTLFormula.from_str("b | X (a U b)")
    assert not spot.check_equiv(f, g).equiv
    del spot


@pytest.mark.docker
def test_spot_incl_2():
    spot = Spot()
    f = LTLFormula.from_str("G F a")
    g = LTLFormula.from_str("F G a")
    assert not spot.inclusion(f, g).right_in_left
    assert spot.inclusion(f, g).left_in_right
    del spot


@pytest.mark.docker
def test_spot_incl_stream():
    spot = Spot()
    a = LTLFormula.from_str("G F a")
    b = LTLFormula.from_str("F G a")
    c = LTLFormula.from_str("X X X G F a")
    d = LTLFormula.from_str("F a")

    def gen():
        for t in [(a, b), (b, c), (c, d), (a, d), (a, d)]:
            yield t

    results = list(spot.inclusion_stream(gen()))
    assert [r.status for r in results] == [
        "only_left_in_right",
        "only_right_in_left",
        "only_right_in_left",
        "only_right_in_left",
        "only_right_in_left",
    ]
    del spot


@pytest.mark.docker
def test_spot_exclusive_word():
    spot = Spot()
    f = LTLFormula.from_str("a U b")
    g = LTLFormula.from_str("b | X (a U b)")
    equiv, word = spot.exclusive_word(f, g)
    assert not equiv.equiv
    assert SymbolicTrace.from_str("!a & !b ; { b }") == word
    del spot


@pytest.mark.docker
def test_spot_sat():
    spot = Spot()
    f = LTLFormula.from_str("a U b")
    s, t = spot.check_sat(f, timeout=TIMEOUT)
    assert s == LTLSatStatus("satisfiable")
    assert t is not None
    del spot


@pytest.mark.docker
def test_spot_unsat():
    spot = Spot()
    f = LTLFormula.from_str("a U b & G ! b")
    s, t = spot.check_sat(f, timeout=TIMEOUT)
    assert s == LTLSatStatus("unsatisfiable")
    assert t is None
    del spot


@pytest.mark.docker
def test_spot_mc_trace_1():
    spot = Spot()
    f = LTLFormula.from_str("F a & F b")
    t = SymbolicTrace.from_str("a & b; cycle{1}", notation="infix", spot=True)
    assert spot.mc_trace(f, t, TIMEOUT).satisfied
    del spot


@pytest.mark.docker
def test_spot_mc_trace_2():
    spot = Spot()
    f = LTLFormula.from_str("F a & F b")
    t = SymbolicTrace.from_str("a; cycle{1}", notation="infix", spot=True)
    assert not spot.mc_trace(f, t, TIMEOUT).satisfied
    del spot


@pytest.mark.docker
def test_spot_mc_trace_3():
    spot = Spot()
    f = LTLFormula.from_str("F a & F b")
    t = SymbolicTrace.from_str("a & b; cycle{1}", notation="infix", spot=True)
    assert spot.mc_trace(f, t, timeout=0) == TraceMCStatus("timeout")
    del spot


@pytest.mark.docker
def test_spot_mc_1():
    spot = Spot()
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
    solution = spot.model_check(
        ToolLTLMCProblem(
            parameters={"timeout": TIMEOUT},
            realizable=True,
            specification=spec,
            circuit=satisfying_system,
        )
    )
    assert solution.status == LTLMCStatus("satisfied")
    assert solution.detailed_status == "SATISFIED"
    assert solution.tool == spot.tool
    assert solution.time_seconds > 0
    del spot


@pytest.mark.docker
def test_spot_mc_2():
    spot = Spot()
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

    solution = spot.model_check(
        ToolLTLMCProblem(
            parameters={"timeout": TIMEOUT},
            realizable=True,
            specification=spec,
            circuit=violating_system,
        )
    )
    assert solution.status == LTLMCStatus("violated")
    assert solution.detailed_status == "VIOLATED"
    assert solution.tool == spot.tool
    assert solution.time_seconds > 0
    assert solution.counterexample == SymbolicTrace.from_str("{ g1 & g2 ; !g1 & !g2 }")
    del spot


@pytest.mark.docker
def test_spot_mc_3():
    spot = Spot()
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
    solution = spot.model_check(
        ToolLTLMCProblem(
            parameters={"timeout": TIMEOUT},
            realizable=False,
            specification=spec,
            circuit=satisfying_system,
        )
    )
    assert solution.status == LTLMCStatus("violated")
    assert solution.detailed_status == "VIOLATED"
    assert solution.tool == spot.tool
    assert solution.time_seconds > 0
    del spot


@pytest.mark.docker
def test_spot_mc_4():
    spot = Spot()
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
    solution = spot.model_check(
        ToolLTLMCProblem(
            parameters={"timeout": 0},
            realizable=True,
            specification=spec,
            circuit=satisfying_system,
        )
    )
    assert solution.status == LTLMCStatus("timeout")
    assert solution.detailed_status.startswith("TIMEOUT")
    assert solution.tool == spot.tool
    del spot


@pytest.mark.docker
def test_spot_mc_5():
    spot = Spot()
    spec = DecompLTLSpec.from_dict(
        {
            "guarantees": [
                "(G ((i0) -> (o4)))",
                "(G (((! (i1)) & (! (i0))) -> (F (((! (o0)) & (! (o2))) & (! (o1))))))",
                "(G ((i1) -> (F (o4))))",
                "(G ((! (o1)) | (! (o0))))",
                "(G (((i0) & (X (i4))) -> (F ((o0) & (o1)))))",
                "(G ((o3) -> (X ((i1) R (((i1) -> (o0)) & ((! (i1)) -> (o1)))))))",
                "(G ((i2) -> (F (o0))))",
                "(G (((o0) & (X ((! (i2)) & (! (o0))))) -> (X ((i2) R (! (o0))))))",
                "(G ((i2) -> (o2)))",
            ],
            "assumptions": [
                "(X (G ((! (o0)) | (((! (i4)) & (! (i3))) U ((! (i4)) & (i3))))))",
                "(G (F (i4)))",
            ],
            "inputs": ["i0", "i1", "i2", "i3", "i4"],
            "outputs": ["o0", "o1", "o2", "o3", "o4"],
        }
    )

    mealy = MealyMachine.from_hoa(
        'HOA: v1\nStates: 2\nStart: 0\nAP: 10 "i0" "i1" "i2" "i3" "i4" "o0" "o1" "o2" "o3" "o4"\nacc-name: all\nAcceptance: 0 t\nproperties: trans-labels explicit-labels state-acc deterministic\ncontrollable-AP: 5 6 7 8 9\n--BODY--\nState: 0\n[!2&5&6&!7&!8&9 | 2&!5&!6&!7&!8&9] 1\nState: 1\n[!4&!5&!6&!7&!8&9 | 4&5&!6&!7&!8&9] 1\n--END--'
    )
    solution = spot.model_check(
        ToolLTLMCProblem(
            parameters={"timeout": 120},
            realizable=True,
            specification=spec,
            mealy_machine=mealy,
        )
    )
    assert solution.status == LTLMCStatus("satisfied")
    assert solution.detailed_status == "SATISFIED"
    assert solution.tool == spot.tool
    assert solution.time_seconds > 0
    del spot


@pytest.mark.docker
def test_spot_aiger_syn():
    spot = Spot()
    spec_dict = {
        "guarantees": [
            "(G ((((! (g_0)) && (! (g_1))) && ((! (g_2)) || (! (g_3)))) || ((((! (g_0)) || (! (g_1))) && (! (g_2))) && (! (g_3)))))",
            "(G ((r_0) -> (F (g_0))))",
            "(G ((r_1) -> (F (g_1))))",
            "(G ((r_2) -> (F (g_2))))",
            "(G ((r_3) -> (F (g_3))))",
        ],
        "inputs": ["r_0", "r_1", "r_2", "r_3", "r_4"],
        "outputs": ["g_0", "g_1", "g_2", "g_3", "g_4"],
    }
    spec = DecompLTLSpec.from_dict(spec_dict)
    syn_solution = spot.synthesize(ToolLTLSynProblem(specification=spec, parameters={}))
    assert syn_solution.circuit is not None
    mc_solution = spot.model_check(
        ToolLTLMCProblem(
            specification=spec,
            circuit=syn_solution.circuit,
            realizable=True,
            parameters={"timeout": TIMEOUT},
        )
    )
    assert mc_solution.status == LTLMCStatus("satisfied")
