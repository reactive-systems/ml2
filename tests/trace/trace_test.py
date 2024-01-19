"""Trace test"""

from ml2.prop import Assignment
from ml2.trace import Trace


def test_trace_str():
    t1 = Trace.from_str("a , ! b ; { b , c ; a }", notation="standard")
    t2 = Trace.from_str("a & ! b ; cycle{ b & c ; a }", notation="spot")
    t3 = Trace.from_str(
        " -> State: 1.1 <-\n    a = TRUE\n    b = FALSE\n  -- Loop starts here\n  -> State: 1.2 <-\n    b = TRUE\n    c=TRUE\n  -> State: 1.3 <-\n    a = TRUE\n",
        notation="nusmv",
    )
    t4 = Trace.from_str("{a,(! b),}\n(\n{b,c,}\n{a,}\n)^w\n", notation="aalta")
    assert t1 == t2
    assert t1 == t3
    assert t1 == t4


def test_trace_complete():
    t = Trace.from_str("a , ! b ; b ; { ! a ; }", notation="standard")
    t.complete_by_predecessor()
    assert t == Trace.from_str("a , ! b ; a , b ; { ! a , b ; ! a , b }", notation="standard")


def test_trace_empty():
    t1 = Trace.from_str("{}", notation="standard")
    t2 = Trace.from_str("cycle{}", notation="spot")
    t3 = Trace.from_str("(\n{,}\n)^w\n", notation="aalta")
    for t in [t1, t2, t3]:
        assert t.prefix == []
        assert t.cycle == [Assignment()]


def test_trace_tokens():
    tokens = ["a", ",", "!", "b", ";", "{", "b", ",", "c", ";", "a", "}"]
    t = Trace.from_tokens(tokens, notation="standard")
    assert t.to_tokens(notation="standard") == tokens
