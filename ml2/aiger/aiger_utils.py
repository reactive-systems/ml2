"""AIGER utils"""


def header_ints_from_str(aiger: str):
    header = aiger.split("\n")[0]
    header_strs = header.split(" ")
    # first position is format identifier string aag
    max_var_id = int(header_strs[1])
    num_inputs = int(header_strs[2])
    num_latches = int(header_strs[3])
    num_outputs = int(header_strs[4])
    num_and_gates = int(header_strs[5])
    return max_var_id, num_inputs, num_latches, num_outputs, num_and_gates


def reconstruct_header_ints(circuit: str, num_inputs: int, num_outputs: int):
    lines = circuit.split("\n")
    max_var = 0
    for line in lines:
        for var in line.split(" "):
            if var.isdigit() and int(var) > max_var:
                max_var = int(var)
    max_var_id = max_var // 2
    lines = lines[num_inputs:]
    num_latches = 0
    for line in lines:
        if (len(line.split(" "))) == 2:
            num_latches += 1
        else:
            break
    lines = lines[num_latches + num_outputs :]
    num_and_gates = 0
    for line in lines:
        if (len(line.split(" "))) == 3:
            num_and_gates += 1
        else:
            break
    return max_var_id, num_inputs, num_latches, num_outputs, num_and_gates
