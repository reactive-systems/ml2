"""BoSy test"""

import pytest

from ml2.ltl import DecompLTLSpec
from ml2.ltl.ltl_syn import LTLSynStatus
from ml2.ltl.ltl_syn.ltl_syn_problem import LTLSynSolution
from ml2.tools.bosy import BoSy
from ml2.tools.ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem, ToolLTLSynSolution

TIMEOUT = 120


@pytest.mark.docker
def test_bosy_1():
    bosy = BoSy()

    unreal_spec = DecompLTLSpec.from_dict(
        {
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = bosy.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT}, specification=unreal_spec, system_format="aiger"
        )
    )
    assert sol.status == LTLSynStatus("unrealizable")


@pytest.mark.docker
def test_bosy_2():
    bosy = BoSy()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G F ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = bosy.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT, "--optimize": ""},
            specification=real_spec,
            system_format="aiger",
        )
    )
    assert sol.status == LTLSynStatus("realizable")


@pytest.mark.docker
def test_bosy_3():
    bosy = BoSy()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G F ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = bosy.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": 0}, specification=real_spec, system_format="aiger"
        )
    )
    assert sol.status == LTLSynStatus("timeout")


@pytest.mark.docker
def test_bosy_4():
    bosy = BoSy()
    assert bosy.assert_identities
    assert bosy.server_version == "2.0"
    assert bosy.functionality == ["FUNCTIONALITY_LTL_AIGER_SYNTHESIS"]


@pytest.mark.docker
def test_bosy_5():
    bosy = BoSy()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G L ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = bosy.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT}, specification=real_spec, system_format="aiger"
        )
    )
    assert sol.status == LTLSynStatus("error")


@pytest.mark.docker
def test_bosy_6():
    bosy = BoSy()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G F ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: LTLSynSolution = bosy.synthesize_spec(spec=real_spec, timeout=TIMEOUT)
    assert sol.status == LTLSynStatus("realizable")


# def test_bosy_6():
#     bosy = Bosy()

#     unreal_spec = DecompLTLSpec.from_dict(
#         {
#             "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
#             "inputs": ["r1", "r2"],
#             "outputs": ["g1", "g2"],
#         }
#     )
#     sol: ToolLTLSynSolution = bosy.synthesize(
#         ToolLTLSynProblem(
#             parameters={"timeout": TIMEOUT}, specification=unreal_spec, system_format="mealy"
#         )
#     )
#     assert sol.status == LTLSynStatus("unrealizable")


# def test_bosy_7():
#     bosy = Bosy()
#     real_spec = DecompLTLSpec.from_dict(
#         {
#             "assumptions": ["G F ! r1"],
#             "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
#             "inputs": ["r1", "r2"],
#             "outputs": ["g1", "g2"],
#         }
#     )
#     sol: ToolLTLSynSolution = bosy.synthesize(
#         ToolLTLSynProblem(
#             parameters={"timeout": TIMEOUT}, specification=real_spec, system_format="mealy"
#         )
#     )
#     assert sol.status == LTLSynStatus("realizable")
