"""NuSMV test"""

import pytest

from ml2.ltl.ltl_mc import LTLMCStatus
from ml2.tools.nusmv import NuSMV

MEM_LIMIT = "2g"


@pytest.mark.docker
def test_nusmv_sat_real(sat_real_ltl_mc_problem):
    nusmv = NuSMV(mem_limit=MEM_LIMIT)
    solution = nusmv.model_check(sat_real_ltl_mc_problem)
    assert solution.status == LTLMCStatus("satisfied")
    assert solution.tool == nusmv.tool
    assert solution.time_seconds > 0


@pytest.mark.docker
def test_nusmv_sat_unreal(sat_unreal_ltl_mc_problem):
    nusmv = NuSMV(mem_limit=MEM_LIMIT)
    solution = nusmv.model_check(sat_unreal_ltl_mc_problem)
    assert solution.status == LTLMCStatus("satisfied")
    assert solution.tool == nusmv.tool
    assert solution.time_seconds > 0


@pytest.mark.docker
def test_nusmv_unsat_real(unsat_real_ltl_mc_problem):
    nusmv = NuSMV(mem_limit=MEM_LIMIT)
    solution = nusmv.model_check(unsat_real_ltl_mc_problem)
    assert solution.status == LTLMCStatus("violated")
    assert solution.counterexample is not None
    assert solution.tool == nusmv.tool
    assert solution.time_seconds > 0


@pytest.mark.docker
def test_nusmv_unsat_unreal(unsat_unreal_ltl_mc_problem):
    nusmv = NuSMV(mem_limit=MEM_LIMIT)
    solution = nusmv.model_check(unsat_unreal_ltl_mc_problem)
    assert solution.status == LTLMCStatus("violated")
    assert solution.counterexample is not None
    assert solution.tool == nusmv.tool
    assert solution.time_seconds > 0


@pytest.mark.docker
def test_nusmv_timeout(timeout_ltl_mc_problem):
    nusmv = NuSMV(mem_limit=MEM_LIMIT)
    solution = nusmv.model_check(timeout_ltl_mc_problem)
    assert solution.status == LTLMCStatus("timeout")
    assert solution.detailed_status.startswith("TIMEOUT")
    assert solution.tool == nusmv.tool
