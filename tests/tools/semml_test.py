"""SemML test"""

import pytest

from ml2.ltl import DecompLTLSpec
from ml2.ltl.ltl_syn import LTLSynStatus
from ml2.tools.ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem, ToolLTLSynSolution
from ml2.tools.semml import Semml

TIMEOUT = 120


@pytest.mark.docker
def test_semml_1():
    semml = Semml()

    unreal_spec = DecompLTLSpec.from_dict(
        {
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = semml.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT}, specification=unreal_spec, system_format="aiger"
        )
    )
    print(sol.detailed_status)
    assert sol.status == LTLSynStatus("unrealizable")


@pytest.mark.docker
def test_semml_2():
    semml = Semml()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G F ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = semml.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT},
            specification=real_spec,
            system_format="aiger",
        )
    )
    assert sol.status == LTLSynStatus("realizable")


@pytest.mark.docker
def test_semml_3():
    semml = Semml()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G F ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = semml.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": 0}, specification=real_spec, system_format="aiger"
        )
    )
    assert sol.status == LTLSynStatus("timeout")


@pytest.mark.docker
def test_semml_4():
    semml = Semml()
    assert semml.assert_identities
    assert semml.server_version == "2.1"
    assert semml.functionality == ["FUNCTIONALITY_LTL_AIGER_SYNTHESIS"]


@pytest.mark.docker
def test_semml_5():
    semml = Semml()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G L ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = semml.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT}, specification=real_spec, system_format="aiger"
        )
    )
    assert sol.status == LTLSynStatus("error")
