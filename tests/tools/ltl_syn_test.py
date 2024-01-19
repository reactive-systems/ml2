"""LTL synthesis gRPC components test"""

from datetime import timedelta

from ml2.aiger import AIGERCircuit
from ml2.grpc.ltl import ltl_mc_pb2, ltl_syn_pb2
from ml2.ltl.ltl_mc.ltl_mc_status import LTLMCStatus
from ml2.ltl.ltl_spec import DecompLTLSpec, LTLAssumptions, LTLGuarantees, LTLSpec
from ml2.ltl.ltl_syn.ltl_syn_status import LTLSynStatus
from ml2.mealy import MealyMachine
from ml2.tools.ltl_tool.tool_ltl_mc_problem import ToolLTLMCSolution
from ml2.tools.ltl_tool.tool_ltl_syn_problem import (
    ToolLTLSynProblem,
    ToolLTLSynSolution,
    ToolNeuralLTLSynSolution,
)


def test_neural_ltl_syn_solution_1():
    synthesis_solution = ToolLTLSynSolution(
        status=LTLSynStatus("realizable"),
        detailed_status="REALIZABLE",
        tool="NeuroSynt",
        time=timedelta(seconds=1.33467870829345),
        circuit=AIGERCircuit.from_str(
            "aag 9 5 1 5 3\n2\n4\n6\n8\n10\n12 18\n1\n1\n1\n0\n16\n14 13 5\n16 15 6\n18 15 7\ni0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4"
        ),
        realizable=True,
    )
    model_checking_solution = ToolLTLMCSolution(
        status=LTLMCStatus("satisfied"),
        detailed_status="SATISFIED",
        tool="nuXmv",
        time=timedelta(seconds=1.33467870829345),
    )
    sol = ToolNeuralLTLSynSolution(
        synthesis_solution=synthesis_solution,
        model_checking_solution=model_checking_solution,
        tool="NeuroSynt",
        time=timedelta(seconds=23.4534345),
    )
    pb2_sol = sol.to_pb2_NeuralLTLSynSolution()
    assert pb2_sol.synthesis_solution.status == ltl_syn_pb2.LTLSYNSTATUS_REALIZABLE
    assert pb2_sol.tool == "NeuroSynt"
    assert pb2_sol.synthesis_solution.tool == "NeuroSynt"
    assert pb2_sol.synthesis_solution.time.seconds == 1
    assert pb2_sol.synthesis_solution.time.nanos == 334679000
    assert pb2_sol.time.seconds == 23
    assert pb2_sol.time.nanos == 453435000
    assert pb2_sol.synthesis_solution.detailed_status == "REALIZABLE"
    assert (
        pb2_sol.synthesis_solution.HasField("realizable") and pb2_sol.synthesis_solution.realizable
    )
    assert not pb2_sol.synthesis_solution.HasField("mealy_machine")
    assert (
        pb2_sol.synthesis_solution.circuit.circuit
        == "aag 9 5 1 5 3\n2\n4\n6\n8\n10\n12 18\n1\n1\n1\n0\n16\n14 13 5\n16 15 6\n18 15 7\ni0 i0\ni1 i1\ni2 i2\ni3 i3\ni4 i4\nl0 l0\no0 o0\no1 o1\no2 o2\no3 o3\no4 o4"
    )
    assert pb2_sol.model_checking_solution.status == ltl_mc_pb2.LTLMCSTATUS_SATISFIED
    assert pb2_sol.model_checking_solution.tool == "nuXmv"
    assert pb2_sol.model_checking_solution.time.seconds == 1
    assert pb2_sol.model_checking_solution.time.nanos == 334679000
    assert pb2_sol.model_checking_solution.detailed_status == "SATISFIED"
    sol_2 = ToolNeuralLTLSynSolution.from_pb2_NeuralLTLSynSolution(pb2_sol)
    assert sol_2.model_checking_solution == sol.model_checking_solution
    assert sol_2.synthesis_solution == sol.synthesis_solution
    assert sol_2.tool == sol.tool
    assert sol_2.time == sol.time
    assert sol_2 == sol


def test_ltl_syn_solution_2():
    sol = ToolLTLSynSolution(
        status=LTLSynStatus("error"),
        detailed_status="ERROR: Sample Error",
        tool="Foo",
        time=timedelta(seconds=0),
        circuit=None,
        realizable=None,
    )
    pb2_sol = sol.to_pb2_LTLSynSolution()
    assert pb2_sol.status == ltl_syn_pb2.LTLSYNSTATUS_ERROR
    assert pb2_sol.tool == "Foo"
    assert pb2_sol.time.seconds == 0
    assert pb2_sol.time.nanos == 0
    assert pb2_sol.detailed_status == "ERROR: Sample Error"
    assert not pb2_sol.HasField("realizable")
    sol_2 = ToolLTLSynSolution.from_pb2_LTLSynSolution(pb2_sol)
    assert sol_2.detailed_status == sol.detailed_status
    assert sol_2.status == sol.status
    assert sol_2.tool == sol.tool
    assert sol_2.time == sol.time
    assert sol_2.circuit == sol.circuit
    assert sol_2.realizable == sol.realizable
    assert sol_2.mealy_machine == sol.mealy_machine
    assert sol_2.system == sol.system
    assert sol_2 == sol


def test_ltl_syn_solution_3():
    sol: ToolLTLSynSolution = ToolLTLSynSolution(
        status=LTLSynStatus("unrealizable"),
        detailed_status="UNREALIZABLE",
        tool="Spot",
        time=timedelta(seconds=1.33467870829345),
        mealy_machine=MealyMachine.from_hoa(
            'HOA: v1\nStates: 4\nStart: 0\nAP: 10 "o0" "o1" "o2" "o3" "o4" "i0" "i1" "i2" "i3" "i4"\nacc-name: all\nAcceptance: 0 t\nproperties: trans-labels explicit-labels state-acc deterministic\ncontrollable-AP: 5 6 7 8 9\n--BODY--\nState: 0\n[0&!1&!5&!6&!7&!8&9] 1\n[!0&!5&!6&!7&!8&9] 2\n[0&1&!5&!6&!7&!8&9] 3\nState: 1\n[0&!5&!6&!7&!8&9] 0\n[!0&!5&!6&!7&!8&9] 3\nState: 2\n[!1&!5&!6&7&!8&9] 0\n[!0&1&!5&!6&7&!8&9] 2\n[0&1&!5&!6&7&!8&9] 3\nState: 3\n[0&!5&!6&!7&!8&9] 0\n[!0&!5&!6&!7&!8&9] 3\n--END--'
        ),
        realizable=False,
    )
    pb2_sol = sol.to_pb2_LTLSynSolution()
    assert pb2_sol.status == ltl_syn_pb2.LTLREALSTATUS_UNREALIZABLE
    assert pb2_sol.tool == "Spot"
    assert pb2_sol.time.seconds == 1
    assert pb2_sol.time.nanos == 334679000
    assert pb2_sol.detailed_status == "UNREALIZABLE"
    assert pb2_sol.HasField("realizable") and not pb2_sol.realizable
    assert not pb2_sol.HasField("circuit")
    assert (
        pb2_sol.mealy_machine.machine
        == 'HOA: v1\nStates: 4\nStart: 0\nAP: 10 "o0" "o1" "o2" "o3" "o4" "i0" "i1" "i2" "i3" "i4"\nacc-name: all\nAcceptance: 0 t\nproperties: trans-labels explicit-labels state-acc deterministic\ncontrollable-AP: 5 6 7 8 9\n--BODY--\nState: 0\n[0&!1&!5&!6&!7&!8&9] 1\n[!0&!5&!6&!7&!8&9] 2\n[0&1&!5&!6&!7&!8&9] 3\nState: 1\n[0&!5&!6&!7&!8&9] 0\n[!0&!5&!6&!7&!8&9] 3\nState: 2\n[!1&!5&!6&7&!8&9] 0\n[!0&1&!5&!6&7&!8&9] 2\n[0&1&!5&!6&7&!8&9] 3\nState: 3\n[0&!5&!6&!7&!8&9] 0\n[!0&!5&!6&!7&!8&9] 3\n--END--'
    )
    sol_2 = ToolLTLSynSolution.from_pb2_LTLSynSolution(pb2_sol)
    assert sol_2.detailed_status == sol.detailed_status
    assert sol_2.status == sol.status
    assert sol_2.tool == sol.tool
    assert sol_2.time == sol.time
    assert sol_2.circuit == sol.circuit
    assert sol_2.realizable == sol.realizable
    # assert sol_2.mealy_machine == sol.mealy_machine
    # assert sol_2.system == sol.system
    # assert sol_2 == sol


def test_ltl_syn_problem_1():
    prob: ToolLTLSynProblem = ToolLTLSynProblem(
        parameters={"foo": "bar", "foo_1": 3},
        decomp_specification=DecompLTLSpec.from_csv_fields(
            {
                "inputs": "i0",
                "outputs": "o0",
                "guarantees": "i0 U o0",
                "assumptions": "F i0,F G i0",
            }
        ),
    )
    pb2_prob = prob.to_pb2_LTLSynProblem()
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
    prob_2 = ToolLTLSynProblem.from_pb2_LTLSynProblem(pb2_prob)
    assert prob_2.parameters == prob.parameters
    assert prob_2.decomp_specification == prob.decomp_specification
    assert prob_2.formula_specification == prob.formula_specification
    assert prob_2 == prob


def test_ltl_syn_problem_2():
    prob: ToolLTLSynProblem = ToolLTLSynProblem(
        parameters={"foo": "bar", "foo_1": 3},
        decomp_specification=DecompLTLSpec(
            assumptions=LTLAssumptions(
                sub_exprs=[
                    LTLSpec(formula="F i0", notation="prefix", inputs=["i0"], outputs=["o0"]),
                    LTLSpec(
                        formula="F G i0",
                        notation="prefix",
                        inputs=["i0"],
                        outputs=["o0"],
                    ),
                ],
                inputs=["i0"],
                outputs=["o0"],
            ),
            guarantees=LTLGuarantees(
                sub_exprs=[
                    LTLSpec(
                        formula="U i0 o0",
                        notation="prefix",
                        inputs=["i0"],
                        outputs=["o0"],
                    )
                ],
                inputs=["i0"],
                outputs=["o0"],
            ),
            inputs=["i0"],
            outputs=["o0"],
        ),
    )
    pb2_prob = prob.to_pb2_LTLSynProblem()
    assert pb2_prob.parameters["foo"] == '"bar"'
    assert pb2_prob.parameters["foo_1"] == "3"
    assert pb2_prob.decomp_specification.assumptions[0].formula == "F i0"
    assert pb2_prob.decomp_specification.assumptions[0].notation == "prefix"
    assert pb2_prob.decomp_specification.assumptions[1].formula == "F G i0"
    assert pb2_prob.decomp_specification.assumptions[1].notation == "prefix"
    assert pb2_prob.decomp_specification.guarantees[0].formula == "U i0 o0"
    assert pb2_prob.decomp_specification.guarantees[0].notation == "prefix"
    assert pb2_prob.decomp_specification.inputs[0] == "i0"
    assert pb2_prob.decomp_specification.outputs[0] == "o0"
    assert not pb2_prob.HasField("formula_specification")
    prob_2 = ToolLTLSynProblem.from_pb2_LTLSynProblem(pb2_prob)
    assert prob_2.parameters == prob.parameters
    assert prob_2.decomp_specification == prob.decomp_specification
    assert prob_2.formula_specification == prob.formula_specification
    assert prob_2 == prob


def test_ltl_syn_problem_3():
    prob: ToolLTLSynProblem = ToolLTLSynProblem(
        parameters={"foo": "bar", "foo_1": 3},
        decomp_specification=DecompLTLSpec(
            assumptions=LTLAssumptions(
                sub_exprs=[
                    LTLSpec(formula="F i0", notation="prefix", inputs=["i0"], outputs=["o0"]),
                    LTLSpec(
                        formula="F G i0",
                        notation="prefix",
                        inputs=["i0"],
                        outputs=["o0"],
                    ),
                ],
                inputs=["i0"],
                outputs=["o0"],
            ),
            guarantees=LTLGuarantees(
                sub_exprs=[
                    LTLSpec(
                        formula="U i0 o0",
                        notation="prefix",
                        inputs=["i0"],
                        outputs=["o0"],
                    )
                ],
                inputs=["i0"],
                outputs=["o0"],
            ),
            inputs=["i0"],
            outputs=["o0"],
        ),
    )
    pb2_prob = prob.to_pb2_LTLSynProblem(notation="infix")
    assert pb2_prob.parameters["foo"] == '"bar"'
    assert pb2_prob.parameters["foo_1"] == "3"
    assert pb2_prob.decomp_specification.assumptions[0].formula == "F ( i0 )"
    assert pb2_prob.decomp_specification.assumptions[0].notation == "infix"
    assert pb2_prob.decomp_specification.assumptions[1].formula == "F ( G ( i0 ) )"
    assert pb2_prob.decomp_specification.assumptions[1].notation == "infix"
    assert pb2_prob.decomp_specification.guarantees[0].formula == "( i0 ) U ( o0 )"
    assert pb2_prob.decomp_specification.guarantees[0].notation == "infix"
    assert pb2_prob.decomp_specification.inputs[0] == "i0"
    assert pb2_prob.decomp_specification.outputs[0] == "o0"
    assert not pb2_prob.HasField("formula_specification")


def test_ltl_syn_problem_4():
    prob: ToolLTLSynProblem = ToolLTLSynProblem(
        parameters={"foo": "bar", "foo_1": 3},
        formula_specification=LTLSpec(
            formula="( F ( i0 ) ) & ( F ( G ( i0 ) ) ) -> ( ( i0 ) U ( o0 ) )",
            inputs=["i0"],
            outputs=["o0"],
        ),
    )
    pb2_prob = prob.to_pb2_LTLSynProblem()
    assert pb2_prob.parameters["foo"] == '"bar"'
    assert pb2_prob.parameters["foo_1"] == "3"
    assert (
        pb2_prob.formula_specification.formula.formula
        == "( ( F ( i0 ) ) & ( F ( G ( i0 ) ) ) ) -> ( ( i0 ) U ( o0 ) )"
    )
    assert pb2_prob.formula_specification.formula.notation == "infix"
    assert pb2_prob.formula_specification.inputs[0] == "i0"
    assert pb2_prob.formula_specification.outputs[0] == "o0"
    assert not pb2_prob.HasField("decomp_specification")
    prob_2 = ToolLTLSynProblem.from_pb2_LTLSynProblem(pb2_prob)
    assert prob_2.parameters == prob.parameters
    assert prob_2.decomp_specification == prob.decomp_specification
    assert prob_2.formula_specification == prob.formula_specification
    assert prob_2 == prob
    pb2_prob = prob.to_pb2_LTLSynProblem(notation="prefix")
    assert pb2_prob.parameters["foo"] == '"bar"'
    assert pb2_prob.parameters["foo_1"] == "3"
    assert pb2_prob.formula_specification.formula.formula == "-> & F i0 F G i0 U i0 o0"
    assert pb2_prob.formula_specification.formula.notation == "prefix"
    assert pb2_prob.formula_specification.inputs[0] == "i0"
    assert pb2_prob.formula_specification.outputs[0] == "o0"
    assert not pb2_prob.HasField("decomp_specification")
