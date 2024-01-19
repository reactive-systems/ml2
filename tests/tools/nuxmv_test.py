"""nuXmv test"""

import pytest

from ml2.ltl.ltl_mc import LTLMCStatus
from ml2.tools.nuxmv import Nuxmv
from ml2.utils.dist_utils import architecture_is_apple_arm

# probably due to emulation nuXmv needs lots of memory and is really slow on M1 chips
if architecture_is_apple_arm():
    MEM_LIMIT = "8g"
    TIMEOUT = 30
else:
    MEM_LIMIT = "2g"
    TIMEOUT = 10


@pytest.mark.docker
def test_nuxmv_metadata():
    nuxmv = Nuxmv(mem_limit=MEM_LIMIT)
    assert nuxmv.assert_identities
    assert nuxmv.server_version == "2.6.0"
    assert nuxmv.functionality == ["FUNCTIONALITY_LTL_AIGER_MODELCHECKING"]


@pytest.mark.docker
def test_nuxmv_sat_real(sat_real_ltl_mc_problem):
    nuxmv = Nuxmv(mem_limit=MEM_LIMIT)
    problem = sat_real_ltl_mc_problem
    problem.parameters["timeout"] = TIMEOUT
    solution = nuxmv.model_check(problem)
    assert solution.status == LTLMCStatus("satisfied")
    assert solution.detailed_status == "SATISFIED"
    assert solution.tool == nuxmv.tool
    assert solution.time_seconds > 0


@pytest.mark.docker
def test_nuxmv_sat_unreal(sat_unreal_ltl_mc_problem):
    nuxmv = Nuxmv(mem_limit=MEM_LIMIT)
    problem = sat_unreal_ltl_mc_problem
    problem.parameters["timeout"] = TIMEOUT
    solution = nuxmv.model_check(problem)
    assert solution.status == LTLMCStatus("satisfied")
    assert solution.detailed_status == "SATISFIED"
    assert solution.tool == nuxmv.tool
    assert solution.time_seconds > 0


@pytest.mark.docker
def test_nuxmv_unsat_real(unsat_real_ltl_mc_problem):
    nuxmv = Nuxmv(mem_limit=MEM_LIMIT)
    problem = unsat_real_ltl_mc_problem
    problem.parameters["timeout"] = TIMEOUT
    solution = nuxmv.model_check(problem)
    assert solution.status == LTLMCStatus("violated")
    assert solution.detailed_status == "VIOLATED"
    assert solution.tool == nuxmv.tool
    assert solution.time_seconds > 0


@pytest.mark.docker
def test_nuxmv_unsat_unreal(unsat_unreal_ltl_mc_problem):
    nuxmv = Nuxmv(mem_limit=MEM_LIMIT)
    problem = unsat_unreal_ltl_mc_problem
    problem.parameters["timeout"] = TIMEOUT
    solution = nuxmv.model_check(problem)
    assert solution.status == LTLMCStatus("violated")
    assert solution.detailed_status == "VIOLATED"
    assert solution.tool == nuxmv.tool
    assert solution.time_seconds > 0


@pytest.mark.docker
def test_nuxmv_invalid(invalid_ltl_mc_problem):
    nuxmv = Nuxmv(mem_limit=MEM_LIMIT)
    problem = invalid_ltl_mc_problem
    problem.parameters["timeout"] = TIMEOUT
    solution = nuxmv.model_check(problem)
    assert solution.status == LTLMCStatus("invalid")
    assert solution.detailed_status == "INVALID:\nERROR: Inputs don't match"
    assert solution.tool == nuxmv.tool
    assert solution.time_seconds > 0


@pytest.mark.docker
def test_nuxmv_timeout(timeout_ltl_mc_problem):
    nuxmv = Nuxmv(mem_limit=MEM_LIMIT)
    solution = nuxmv.model_check(timeout_ltl_mc_problem)
    assert solution.status == LTLMCStatus("timeout")
    assert solution.detailed_status.startswith("TIMEOUT")
    assert solution.tool == nuxmv.tool
