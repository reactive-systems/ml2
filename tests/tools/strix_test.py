"""Strix test"""

import pytest

from ml2.ltl import DecompLTLSpec
from ml2.ltl.ltl_syn import LTLSynStatus
from ml2.ltl.ltl_syn.ltl_syn_problem import LTLSynSolution
from ml2.tools.ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem, ToolLTLSynSolution
from ml2.tools.strix import Strix

TIMEOUT = 120


@pytest.mark.docker
def test_strix_1():
    strix = Strix()

    unreal_spec = DecompLTLSpec.from_dict(
        {
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = strix.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT}, specification=unreal_spec, system_format="aiger"
        )
    )
    assert sol.status == LTLSynStatus("unrealizable")


@pytest.mark.docker
def test_strix_2():
    strix = Strix()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G F ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = strix.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT, "--minimize": "both"},
            specification=real_spec,
            system_format="aiger",
        )
    )
    assert sol.status == LTLSynStatus("realizable")


@pytest.mark.docker
def test_strix_3():
    strix = Strix()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G F ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = strix.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": 0}, specification=real_spec, system_format="aiger"
        )
    )
    assert sol.status == LTLSynStatus("timeout")


@pytest.mark.docker
def test_strix_4():
    strix = Strix()
    assert strix.assert_identities
    assert strix.server_version == "2.1"
    assert strix.functionality == [
        "FUNCTIONALITY_LTL_AIGER_SYNTHESIS",
        "FUNCTIONALITY_LTL_MEALY_SYNTHESIS",
    ]


@pytest.mark.docker
def test_strix_5():
    strix = Strix()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G L ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = strix.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT}, specification=real_spec, system_format="aiger"
        )
    )
    assert sol.status == LTLSynStatus("error")


@pytest.mark.docker
def test_strix_6():
    strix = Strix()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G F ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: LTLSynSolution = strix.synthesize_spec(spec=real_spec, timeout=TIMEOUT)
    assert sol.status == LTLSynStatus("realizable")


def test_strix_7():
    strix = Strix()

    unreal_spec = DecompLTLSpec.from_dict(
        {
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = strix.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT}, specification=unreal_spec, system_format="mealy"
        )
    )
    assert sol.status == LTLSynStatus("unrealizable")


def test_strix_8():
    strix = Strix()
    real_spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G F ! r1"],
            "guarantees": ["G (r1 -> X (! g2 U g1))", "G (r2 -> F g2)", "G ! (g1 & g2)"],
            "inputs": ["r1", "r2"],
            "outputs": ["g1", "g2"],
        }
    )
    sol: ToolLTLSynSolution = strix.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": TIMEOUT}, specification=real_spec, system_format="mealy"
        )
    )
    assert sol.status == LTLSynStatus("realizable")
