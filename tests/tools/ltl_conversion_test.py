"""LTL conversion test"""

from datetime import timedelta

from ml2.ltl.ltl_spec import DecompLTLSpec
from ml2.tools.ltl_tool.tool_ltl_conversion import (
    ToolLTLConversionRequest,
    ToolLTLConversionResponse,
)


def test_ltl_conversion_response_1():
    sol = ToolLTLConversionResponse(
        error="",
        specification=DecompLTLSpec.from_csv_fields(
            {
                "inputs": "i0",
                "outputs": "o0",
                "guarantees": "i0 U o0",
                "assumptions": "F i0,F G i0",
                "name": "test_name",
                "semantics": "Mealy",
            }
        ),
        tool="SyFCo",
        time=timedelta(seconds=1.33467870829345),
    )
    pb2_sol = sol.to_pb2_ConvertTLSFToSpecResponse()
    assert pb2_sol.tool == "SyFCo"
    assert pb2_sol.time.seconds == 1
    assert pb2_sol.time.nanos == 334679000
    assert pb2_sol.error == ""
    assert pb2_sol.specification.assumption_properties.sub_exprs[0].formula.formula == "F i0"
    assert pb2_sol.specification.assumption_properties.sub_exprs[0].formula.notation == "infix"
    assert pb2_sol.specification.assumption_properties.sub_exprs[1].formula.formula == "F G i0"
    assert pb2_sol.specification.assumption_properties.sub_exprs[1].formula.notation == "infix"
    assert pb2_sol.specification.guarantee_properties.sub_exprs[0].formula.formula == "i0 U o0"
    assert pb2_sol.specification.guarantee_properties.sub_exprs[0].formula.notation == "infix"
    assert pb2_sol.specification.inputs[0] == "i0"
    assert pb2_sol.specification.outputs[0] == "o0"
    assert pb2_sol.specification.semantics == "Mealy"
    assert pb2_sol.specification.name == "test_name"
    sol_2 = ToolLTLConversionResponse.from_pb2_ConvertTLSFToSpecResponse(pb2_sol)
    assert sol_2.error == sol.error
    assert sol_2.tool == sol.tool
    assert sol_2.time == sol.time
    assert sol_2.decomp_specification == sol.decomp_specification
    assert sol_2.formula_specification == sol.formula_specification
    assert sol_2.specification == sol.specification
    assert sol_2 == sol


def test_ltl_conversion_response_2():
    sol = ToolLTLConversionResponse(
        error="ERROR: Sample Error",
        tool="Foo",
        time=timedelta(seconds=0),
    )
    pb2_sol = sol.to_pb2_ConvertTLSFToSpecResponse()
    assert pb2_sol.tool == "Foo"
    assert pb2_sol.time.seconds == 0
    assert pb2_sol.time.nanos == 0
    assert pb2_sol.error == "ERROR: Sample Error"
    sol_2 = ToolLTLConversionResponse.from_pb2_ConvertTLSFToSpecResponse(pb2_sol)
    assert sol_2.error == sol.error
    assert sol_2.tool == sol.tool
    assert sol_2.time == sol.time
    assert sol_2 == sol


def test_ltl_conversion_1():
    prob = ToolLTLConversionRequest(
        parameters={"foo": "bar", "foo_1": 3},
        tlsf_string='INFO {\n  TITLE:       "Converted TSL Specification: ActionConverter"\n  DESCRIPTION: "TSL specification, which has been converted to TLSF."\n  SEMANTICS:   Mealy\n  TARGET:      Mealy\n}\nMAIN {\n  INPUTS {\n    p0p0iscockpitmode0gamemode;\n    p0p0isscoremode0gamemode;\n    p0p0gt0accz0f1dresetthreshhold1b;\n    p0p0gt0f1dabs0accz1b0f1dshotthreshhold1b;\n    p0p0norotation0gyrox0gyroy0gyroz;\n  }\n\n  OUTPUTS {\n    u0gamestart0gamestart;\n    u0gamestart0f1dbot1b;\n    u0gamestart0f1dtop1b;\n    u0shot0shot;\n    u0shot0f1dbot1b;\n    u0shot0f1dtop1b;\n  }\n\n  GUARANTEE {\n    (G ((! (((u0gamestart0f1dbot1b) && (! ((u0gamestart0f1dtop1b) || (u0gamestart0gamestart)))) <-> ((! (((u0gamestart0f1dtop1b) && (! (u0gamestart0gamestart))) <-> ((u0gamestart0gamestart) && (! (u0gamestart0f1dtop1b))))) && (! (u0gamestart0f1dbot1b))))) && (! (((u0shot0f1dbot1b) && (! ((u0shot0f1dtop1b) || (u0shot0shot)))) <-> ((! (((u0shot0f1dtop1b) && (! (u0shot0shot))) <-> ((u0shot0shot) && (! (u0shot0f1dtop1b))))) && (! (u0shot0f1dbot1b))))))) && ((((G ((u0gamestart0f1dtop1b) || (u0gamestart0f1dbot1b))) && (G ((u0shot0f1dtop1b) || (u0shot0f1dbot1b)))) && (G ((((p0p0isscoremode0gamemode) && (p0p0gt0accz0f1dresetthreshhold1b)) && (p0p0norotation0gyrox0gyroy0gyroz)) <-> (u0gamestart0f1dtop1b)))) && (G ((((p0p0iscockpitmode0gamemode) && (p0p0gt0f1dabs0accz1b0f1dshotthreshhold1b)) && (p0p0norotation0gyrox0gyroy0gyroz)) <-> (u0shot0f1dtop1b))));\n  }\n}\n',
    )
    pb2_prob = prob.to_pb2_ConvertTLSFToSpecRequest()
    assert pb2_prob.parameters["foo"] == '"bar"'
    assert pb2_prob.parameters["foo_1"] == "3"
    assert (
        pb2_prob.tlsf.tlsf
        == 'INFO {\n  TITLE:       "Converted TSL Specification: ActionConverter"\n  DESCRIPTION: "TSL specification, which has been converted to TLSF."\n  SEMANTICS:   Mealy\n  TARGET:      Mealy\n}\nMAIN {\n  INPUTS {\n    p0p0iscockpitmode0gamemode;\n    p0p0isscoremode0gamemode;\n    p0p0gt0accz0f1dresetthreshhold1b;\n    p0p0gt0f1dabs0accz1b0f1dshotthreshhold1b;\n    p0p0norotation0gyrox0gyroy0gyroz;\n  }\n\n  OUTPUTS {\n    u0gamestart0gamestart;\n    u0gamestart0f1dbot1b;\n    u0gamestart0f1dtop1b;\n    u0shot0shot;\n    u0shot0f1dbot1b;\n    u0shot0f1dtop1b;\n  }\n\n  GUARANTEE {\n    (G ((! (((u0gamestart0f1dbot1b) && (! ((u0gamestart0f1dtop1b) || (u0gamestart0gamestart)))) <-> ((! (((u0gamestart0f1dtop1b) && (! (u0gamestart0gamestart))) <-> ((u0gamestart0gamestart) && (! (u0gamestart0f1dtop1b))))) && (! (u0gamestart0f1dbot1b))))) && (! (((u0shot0f1dbot1b) && (! ((u0shot0f1dtop1b) || (u0shot0shot)))) <-> ((! (((u0shot0f1dtop1b) && (! (u0shot0shot))) <-> ((u0shot0shot) && (! (u0shot0f1dtop1b))))) && (! (u0shot0f1dbot1b))))))) && ((((G ((u0gamestart0f1dtop1b) || (u0gamestart0f1dbot1b))) && (G ((u0shot0f1dtop1b) || (u0shot0f1dbot1b)))) && (G ((((p0p0isscoremode0gamemode) && (p0p0gt0accz0f1dresetthreshhold1b)) && (p0p0norotation0gyrox0gyroy0gyroz)) <-> (u0gamestart0f1dtop1b)))) && (G ((((p0p0iscockpitmode0gamemode) && (p0p0gt0f1dabs0accz1b0f1dshotthreshhold1b)) && (p0p0norotation0gyrox0gyroy0gyroz)) <-> (u0shot0f1dtop1b))));\n  }\n}\n'
    )
    prob_2 = ToolLTLConversionRequest.from_pb2_ConvertTLSFToSpecRequest(pb2_prob)
    assert prob_2.parameters == prob.parameters
    assert prob_2.tlsf_string == prob.tlsf_string
    assert prob_2 == prob
