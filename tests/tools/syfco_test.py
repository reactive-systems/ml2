"""Syfco test"""

import pytest

from ml2.ltl import DecompLTLSpec
from ml2.tools.syfco import Syfco

TIMEOUT = 10


@pytest.mark.docker
def test_syfco_1():
    syfco = Syfco()
    assert syfco.assert_identities
    assert syfco.server_version == "1.2.1.3"
    assert syfco.functionality == ["FUNCTIONALITY_TLSF_TO_SPEC"]


@pytest.mark.docker
def test_syfco_2():
    syfco = Syfco()
    file_str = 'INFO {\n  TITLE:       "Converted TSL Specification: ActionConverter"\n  DESCRIPTION: "TSL specification, which has been converted to TLSF."\n  SEMANTICS:   Mealy\n  TARGET:      Mealy\n}\nMAIN {\n  INPUTS {\n    p0p0iscockpitmode0gamemode;\n    p0p0isscoremode0gamemode;\n    p0p0gt0accz0f1dresetthreshhold1b;\n    p0p0gt0f1dabs0accz1b0f1dshotthreshhold1b;\n    p0p0norotation0gyrox0gyroy0gyroz;\n  }\n\n  OUTPUTS {\n    u0gamestart0gamestart;\n    u0gamestart0f1dbot1b;\n    u0gamestart0f1dtop1b;\n    u0shot0shot;\n    u0shot0f1dbot1b;\n    u0shot0f1dtop1b;\n  }\n\n  GUARANTEE {\n    (G ((! (((u0gamestart0f1dbot1b) && (! ((u0gamestart0f1dtop1b) || (u0gamestart0gamestart)))) <-> ((! (((u0gamestart0f1dtop1b) && (! (u0gamestart0gamestart))) <-> ((u0gamestart0gamestart) && (! (u0gamestart0f1dtop1b))))) && (! (u0gamestart0f1dbot1b))))) && (! (((u0shot0f1dbot1b) && (! ((u0shot0f1dtop1b) || (u0shot0shot)))) <-> ((! (((u0shot0f1dtop1b) && (! (u0shot0shot))) <-> ((u0shot0shot) && (! (u0shot0f1dtop1b))))) && (! (u0shot0f1dbot1b))))))) && ((((G ((u0gamestart0f1dtop1b) || (u0gamestart0f1dbot1b))) && (G ((u0shot0f1dtop1b) || (u0shot0f1dbot1b)))) && (G ((((p0p0isscoremode0gamemode) && (p0p0gt0accz0f1dresetthreshhold1b)) && (p0p0norotation0gyrox0gyroy0gyroz)) <-> (u0gamestart0f1dtop1b)))) && (G ((((p0p0iscockpitmode0gamemode) && (p0p0gt0f1dabs0accz1b0f1dshotthreshhold1b)) && (p0p0norotation0gyrox0gyroy0gyroz)) <-> (u0shot0f1dtop1b))));\n  }\n}\n'
    response = syfco.from_tlsf_str(file_str)
    spec = DecompLTLSpec.from_csv_fields(
        {
            "inputs": "p0p0norotation0gyrox0gyroy0gyroz,p0p0gt0f1dabs0accz1b0f1dshotthreshhold1b,p0p0gt0accz0f1dresetthreshhold1b,p0p0isscoremode0gamemode,p0p0iscockpitmode0gamemode",
            "outputs": "u0shot0f1dtop1b,u0shot0f1dbot1b,u0shot0shot,u0gamestart0f1dtop1b,u0gamestart0f1dbot1b,u0gamestart0gamestart",
            "assumptions": "",
            "guarantees": "(G (((((u0gamestart0f1dbot1b) & (! (u0gamestart0f1dtop1b))) & (! (u0gamestart0gamestart))) <-> ((((u0gamestart0f1dtop1b) & (! (u0gamestart0gamestart))) <-> ((u0gamestart0gamestart) & (! (u0gamestart0f1dtop1b)))) | (u0gamestart0f1dbot1b))) & ((((u0shot0f1dbot1b) & (! (u0shot0f1dtop1b))) & (! (u0shot0shot))) <-> ((((u0shot0f1dtop1b) & (! (u0shot0shot))) <-> ((u0shot0shot) & (! (u0shot0f1dtop1b)))) | (u0shot0f1dbot1b))))),(G ((u0gamestart0f1dtop1b) | (u0gamestart0f1dbot1b))),(G ((u0shot0f1dtop1b) | (u0shot0f1dbot1b))),(G ((((p0p0isscoremode0gamemode) & (p0p0gt0accz0f1dresetthreshhold1b)) & (p0p0norotation0gyrox0gyroy0gyroz)) <-> (u0gamestart0f1dtop1b))),(G ((((p0p0iscockpitmode0gamemode) & (p0p0gt0f1dabs0accz1b0f1dshotthreshhold1b)) & (p0p0norotation0gyrox0gyroy0gyroz)) <-> (u0shot0f1dtop1b)))",
            "semantics": "mealy",
        }
    )
    assert response == spec
