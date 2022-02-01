"""Collection of useful functions related to data processing"""


def int_to_abbrev_str(n: int):
    """Given an integer returns an abbreviated string representing the integer, e.g., '100K' given 100000"""
    if n > 0 and n % 10 ** 6 == 0:
        return f"{n // 10**6}M"
    elif n > 0 and n % 10 ** 3 == 0:
        return f"{n // 10**3}K"
    else:
        return f"{n}"


def from_csv_str(s: str):
    """Escapes a string that is read from a csv file"""
    return s.replace("\\n", "\n")


def to_csv_str(s: str):
    """Escapes a string that is supposed to be written to a csv file"""
    return s.replace("\n", "\\n")
