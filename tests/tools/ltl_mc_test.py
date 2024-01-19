"""LTL model checking gRPC components test"""

from datetime import timedelta

from ml2.aiger import AIGERCircuit
from ml2.grpc.ltl import ltl_mc_pb2
from ml2.ltl.ltl_mc.ltl_mc_status import LTLMCStatus
from ml2.ltl.ltl_spec import DecompLTLSpec
from ml2.mealy import MealyMachine
from ml2.tools.ltl_tool.tool_ltl_mc_problem import ToolLTLMCProblem, ToolLTLMCSolution


def test_ltl_mc_solution_1():
    sol = ToolLTLMCSolution(
        status=LTLMCStatus("satisfied"),
        detailed_status="SATISFIED",
        tool="nuXmv",
        time=timedelta(seconds=1.33467870829345),
    )
    pb2_sol = sol.to_pb2_LTLMCSolution()
    assert pb2_sol.status == ltl_mc_pb2.LTLMCSTATUS_SATISFIED
    assert pb2_sol.tool == "nuXmv"
    assert pb2_sol.time.seconds == 1
    assert pb2_sol.time.nanos == 334679000
    assert pb2_sol.detailed_status == "SATISFIED"
    sol_2 = ToolLTLMCSolution.from_pb2_LTLMCSolution(pb2_sol)
    assert sol_2.detailed_status == sol.detailed_status
    assert sol_2.status == sol.status
    assert sol_2.tool == sol.tool
    assert sol_2.time == sol.time
    assert sol_2 == sol
    # TODO symbolic trace


def test_ltl_mc_solution_2():
    sol = ToolLTLMCSolution(
        status=LTLMCStatus("error"),
        detailed_status="ERROR: Sample Error",
        tool="Foo",
        time=timedelta(seconds=0),
    )
    pb2_sol = sol.to_pb2_LTLMCSolution()
    assert pb2_sol.status == ltl_mc_pb2.LTLMCSTATUS_ERROR
    assert pb2_sol.tool == "Foo"
    assert pb2_sol.time.seconds == 0
    assert pb2_sol.time.nanos == 0
    assert pb2_sol.detailed_status == "ERROR: Sample Error"
    sol_2 = ToolLTLMCSolution.from_pb2_LTLMCSolution(pb2_sol)
    assert sol_2.detailed_status == sol.detailed_status
    assert sol_2.status == sol.status
    assert sol_2.tool == sol.tool
    assert sol_2.time == sol.time
    assert sol_2 == sol


def test_ltl_mc_problem_1():
    prob = ToolLTLMCProblem(
        parameters={"foo": "bar", "foo_1": 3},
        realizable=True,
        decomp_specification=DecompLTLSpec.from_csv_fields(
            {
                "inputs": "i0",
                "outputs": "o0",
                "guarantees": "i0 U o0",
                "assumptions": "F i0,F G i0",
            }
        ),
        circuit=AIGERCircuit.from_str(
            "aag 9 5 1 5 3\n2\n4\n6\n8\n10\n12 18\n1\n1\n1\n0\n16\n14 13 5\n16 15 6\n18 15 7\ni0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4"
        ),
    )
    pb2_prob = prob.to_pb2_LTLMCProblem()
    assert pb2_prob.parameters["foo"] == '"bar"'
    assert pb2_prob.parameters["foo_1"] == "3"
    assert pb2_prob.decomp_specification.assumptions[0].formula == "F i0"
    assert pb2_prob.decomp_specification.assumptions[0].notation == "infix"
    assert pb2_prob.decomp_specification.assumptions[1].formula == "F G i0"
    assert pb2_prob.decomp_specification.assumptions[1].notation == "infix"
    assert pb2_prob.decomp_specification.guarantees[0].formula == "i0 U o0"
    assert pb2_prob.decomp_specification.guarantees[0].notation == "infix"
    assert pb2_prob.decomp_specification.inputs[0] == "i0"
    assert pb2_prob.decomp_specification.outputs[0] == "o0"
    assert not pb2_prob.HasField("formula_specification")
    assert not pb2_prob.HasField("mealy_machine")
    assert (
        pb2_prob.circuit.circuit
        == "aag 9 5 1 5 3\n2\n4\n6\n8\n10\n12 18\n1\n1\n1\n0\n16\n14 13 5\n16 15 6\n18 15 7\ni0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4"
    )
    prob_2 = ToolLTLMCProblem.from_pb2_LTLMCProblem(pb2_prob)
    assert prob_2.parameters == prob.parameters
    assert prob_2.decomp_specification == prob.decomp_specification
    assert prob_2.formula_specification == prob.formula_specification
    assert prob_2.circuit == prob.circuit
    assert prob_2.mealy_machine == prob.mealy_machine
    assert prob_2.system == prob.system
    assert prob_2 == prob


def test_ltl_mc_problem_2():
    prob = ToolLTLMCProblem(
        parameters={"foo": "bar", "foo_1": 3},
        realizable=False,
        decomp_specification=DecompLTLSpec.from_csv_fields(
            {
                "inputs": "i0,i1",
                "outputs": "o0",
                "guarantees": "i0 U o0",
                "assumptions": "F i0,F G i0",
            }
        ),
        mealy_machine=MealyMachine.from_hoa(
            'HOA: v1\nStates: 4\nStart: 0\nAP: 10 "o0" "o1" "o2" "o3" "o4" "i0" "i1" "i2" "i3" "i4"\nacc-name: all\nAcceptance: 0 t\nproperties: trans-labels explicit-labels state-acc deterministic\ncontrollable-AP: 5 6 7 8 9\n--BODY--\nState: 0\n[0&!1&!5&!6&!7&!8&9] 1\n[!0&!5&!6&!7&!8&9] 2\n[0&1&!5&!6&!7&!8&9] 3\nState: 1\n[0&!5&!6&!7&!8&9] 0\n[!0&!5&!6&!7&!8&9] 3\nState: 2\n[!1&!5&!6&7&!8&9] 0\n[!0&1&!5&!6&7&!8&9] 2\n[0&1&!5&!6&7&!8&9] 3\nState: 3\n[0&!5&!6&!7&!8&9] 0\n[!0&!5&!6&!7&!8&9] 3\n--END--'
        ),
    )
    pb2_prob = prob.to_pb2_LTLMCProblem()
    assert pb2_prob.parameters["foo"] == '"bar"'
    assert pb2_prob.parameters["foo_1"] == "3"
    assert pb2_prob.decomp_specification.assumptions[0].formula == "F i0"
    assert pb2_prob.decomp_specification.assumptions[0].notation == "infix"
    assert pb2_prob.decomp_specification.assumptions[1].formula == "F G i0"
    assert pb2_prob.decomp_specification.assumptions[1].notation == "infix"
    assert pb2_prob.decomp_specification.guarantees[0].formula == "i0 U o0"
    assert pb2_prob.decomp_specification.guarantees[0].notation == "infix"
    assert pb2_prob.decomp_specification.inputs[0] == "i0"
    assert pb2_prob.decomp_specification.inputs[1] == "i1"
    assert pb2_prob.decomp_specification.outputs[0] == "o0"
    assert not pb2_prob.HasField("formula_specification")
    assert not pb2_prob.HasField("circuit")
    assert (
        pb2_prob.mealy_machine.machine
        == 'HOA: v1\nStates: 4\nStart: 0\nAP: 10 "o0" "o1" "o2" "o3" "o4" "i0" "i1" "i2" "i3" "i4"\nacc-name: all\nAcceptance: 0 t\nproperties: trans-labels explicit-labels state-acc deterministic\ncontrollable-AP: 5 6 7 8 9\n--BODY--\nState: 0\n[0&!1&!5&!6&!7&!8&9] 1\n[!0&!5&!6&!7&!8&9] 2\n[0&1&!5&!6&!7&!8&9] 3\nState: 1\n[0&!5&!6&!7&!8&9] 0\n[!0&!5&!6&!7&!8&9] 3\nState: 2\n[!1&!5&!6&7&!8&9] 0\n[!0&1&!5&!6&7&!8&9] 2\n[0&1&!5&!6&7&!8&9] 3\nState: 3\n[0&!5&!6&!7&!8&9] 0\n[!0&!5&!6&!7&!8&9] 3\n--END--'
    )
    prob_2 = ToolLTLMCProblem.from_pb2_LTLMCProblem(pb2_prob)
    assert prob_2.parameters == prob.parameters
    assert prob_2.decomp_specification is not None
    assert prob_2.decomp_specification == prob.decomp_specification
    assert prob_2.decomp_specification.inputs == ["i0", "i1"]
    assert type(prob_2.decomp_specification.inputs) == list
    assert prob_2.formula_specification == prob.formula_specification
    assert prob_2.circuit == prob.circuit
    # assert prob_2.mealy_machine == prob.mealy_machine
    # assert prob_2.system == prob.system
    # assert prob_2 == prob
