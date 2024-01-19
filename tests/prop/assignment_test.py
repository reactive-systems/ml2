"""Assignment test"""


from ml2.prop import Assignment


def test_assignment_dict():
    assign = Assignment()
    assign["a"] = True
    assert "a" in assign
    assert assign["a"]
    assign["a"] = False
    assert not assign["a"]


def test_assignment_str():
    a1 = Assignment.from_str("a1,b0,c1", delimiter=",", value_type="num")
    a2 = Assignment.from_str(
        "a=True,b=False,c=True", assign_op="=", delimiter=",", value_type="bool"
    )
    a3 = Assignment.from_str("a, ! b, c", not_op="!", delimiter=",")
    assert a1 == a2
    assert a1 == a3
    assert a1.to_str(assign_op="=", delimiter=",", value_type="bool") == "a=True,b=False,c=True"


def test_assignment_tokens():
    a1 = Assignment.from_tokens(["a", "1", "b", "0", "c", "1"], value_type="num")
    a2 = Assignment.from_tokens(["a", "True", "b", "False", "c", "True"], value_type="bool")
    a3 = Assignment.from_tokens(["a", ",", "!", "b", ",", "c"], not_op="!", delimiter=",")
    assert a1 == a2
    assert a1 == a3
    assert a1.to_tokens(value_type="bool") == ["a", "True", "b", "False", "c", "True"]


def test_assignment_empty():
    assert Assignment.from_str(" ") == Assignment()
