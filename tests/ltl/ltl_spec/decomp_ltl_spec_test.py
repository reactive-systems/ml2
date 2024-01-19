"""Decomposed LTL specification test"""

import json
import os

import pytest

from ml2.ltl.ltl_spec import DecompLTLSpec, LTLAssumptions, LTLGuarantees, LTLSpec


@pytest.fixture()
def arbiter_guarantees():
    res_1 = LTLSpec.from_str("G (i1 -> F o1)")
    res_2 = LTLSpec.from_str("G (i2 -> F o2)")
    mut = LTLSpec.from_str("G ! (o1 & o2)")
    return LTLGuarantees(
        [res_1, res_2, mut],
        inputs=["i1", "i2"],
        outputs=["o1", "o2"],
    )


@pytest.fixture()
def decomp_arbiter_spec(arbiter_guarantees):
    assumptions = LTLAssumptions()
    return DecompLTLSpec(
        assumptions=assumptions,
        guarantees=arbiter_guarantees,
        inputs=["i1", "i2"],
        outputs=["o1", "o2"],
    )


def test_decomp_arbiter_spec(decomp_arbiter_spec):
    assert len(decomp_arbiter_spec.assumptions) == 0
    assert decomp_arbiter_spec.assumption_str(notation="infix") == ""
    assert len(decomp_arbiter_spec.guarantees) == 3
    assert (
        decomp_arbiter_spec.guarantee_str(notation="infix")
        == "( G (i1 -> F o1) ) & ( G (i2 -> F o2) ) & ( G ! (o1 & o2) )"
    )


@pytest.fixture()
def decomp_prio_arbiter_spec():
    no_prio = LTLSpec.from_str("G F ! i1")
    res_1 = LTLSpec.from_str("G (i1 -> X (! o2 U o1))")
    res_2 = LTLSpec.from_str("G (i2 -> F o2)")
    mut = LTLSpec.from_str("G ! (o1 & o2)")
    assumptions = LTLAssumptions([no_prio], inputs=["i1"])
    guarantees = LTLGuarantees(
        [res_1, res_2, mut],
        inputs=["i1", "i2"],
        outputs=["o1", "o2"],
    )
    return DecompLTLSpec(
        assumptions=assumptions,
        guarantees=guarantees,
        inputs=["i1", "i2"],
        outputs=["o1", "o2"],
        semantics="mealy",
    )


def test_decomp_prio_arbiter_spec(decomp_prio_arbiter_spec):
    assert len(decomp_prio_arbiter_spec.assumptions) == 1
    assert decomp_prio_arbiter_spec.assumption_str(notation="infix") == "( G F ! i1 )"
    assert len(decomp_prio_arbiter_spec.guarantees) == 3
    assert (
        decomp_prio_arbiter_spec.guarantee_str(notation="infix")
        == "( G (i1 -> X (! o2 U o1)) ) & ( G (i2 -> F o2) ) & ( G ! (o1 & o2) )"
    )


def test_to_bosy_file(decomp_prio_arbiter_spec, tmp_path):
    filepath = os.path.join(tmp_path, "prio_arbiter.json")
    decomp_prio_arbiter_spec.to_bosy_file(filepath=filepath)
    with open(filepath, "r") as bosy_file:
        bosy_json = json.load(bosy_file)
    assert bosy_json == {
        "semantics": "mealy",
        "inputs": ["i1", "i2"],
        "outputs": ["o1", "o2"],
        "assumptions": ["G F ! i1"],
        "guarantees": ["G (i1 -> X (! o2 U o1))", "G (i2 -> F o2)", "G ! (o1 & o2)"],
    }
