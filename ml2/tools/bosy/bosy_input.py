"""BoSy input"""

BOSY_INPUT_TEMPL = """{{
    "semantics": "{semantics}",
    "inputs": [{inputs}],
    "outputs": [{outputs}],
    "assumptions": [{assumptions}],
    "guarantees": [{guarantees}]
}}"""


def format_bosy_input(guarantees, inputs, outputs, assumptions=None, semantics="mealy"):
    if inputs == []:
        # BoSy requires at least one input
        # TODO report BoSy bug
        inputs = ["i_default"]
    inputs_str = ",".join([f'"{i}"' for i in inputs])
    outputs_str = ",".join([f'"{o}"' for o in outputs])
    assumptions_str = (
        ",\n".join([f'"{assumption}"' for assumption in assumptions]) if assumptions else ""
    )
    guarantees_str = ",\n".join([f'"{guarantee}"' for guarantee in guarantees])
    return BOSY_INPUT_TEMPL.format(
        semantics=semantics,
        inputs=inputs_str,
        outputs=outputs_str,
        assumptions=f"{assumptions_str}",
        guarantees=f"{guarantees_str}",
    )
