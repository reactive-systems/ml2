"""BoSy utilities"""

from .bosy_input import BOSY_INPUT_TEMPL
from .bosy_wrapper import bosy_wrapper_str

COMPILE_TIMEOUT = 300


def bosy_compile(bosy_path):
    # call BoSy with abitrary input to compile BoSy
    print("Compiling BoSy ...")
    bosy_semantics = "mealy"
    inputs_str = '"i"'
    outputs_str = '"o"'
    formula_str = "i U o"
    input_str = BOSY_INPUT_TEMPL.format(
        semantics=bosy_semantics,
        inputs=inputs_str,
        outputs=outputs_str,
        assumptions="",
        guarantees=f'"{formula_str}"',
    )
    bosy_wrapper_str(input_str, bosy_path, COMPILE_TIMEOUT, "/tmp")
    print("Finished compiling BoSy")


if __name__ == "__main__":
    bosy_compile("/bosy/bosy.sh")
