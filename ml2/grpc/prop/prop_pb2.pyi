"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class PropSatProblem(google.protobuf.message.Message):
    """propositional satisfiability problem"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FORMULA_FIELD_NUMBER: builtins.int
    TIMEOUT_FIELD_NUMBER: builtins.int
    formula: builtins.str
    timeout: builtins.float
    def __init__(
        self,
        *,
        formula: builtins.str = ...,
        timeout: builtins.float = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["formula", b"formula", "timeout", b"timeout"]) -> None: ...

global___PropSatProblem = PropSatProblem

class PropSatSolution(google.protobuf.message.Message):
    """solution to a propositional satisfiability problem"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class AssignmentEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.int
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.int = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    STATUS_FIELD_NUMBER: builtins.int
    ASSIGNMENT_FIELD_NUMBER: builtins.int
    status: builtins.str
    @property
    def assignment(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.int]: ...
    def __init__(
        self,
        *,
        status: builtins.str = ...,
        assignment: collections.abc.Mapping[builtins.str, builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["assignment", b"assignment", "status", b"status"]) -> None: ...

global___PropSatSolution = PropSatSolution

class Clause(google.protobuf.message.Message):
    """clause (finite disjunction of literals)"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    LITERALS_FIELD_NUMBER: builtins.int
    @property
    def literals(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        literals: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["literals", b"literals"]) -> None: ...

global___Clause = Clause

class ResClause(google.protobuf.message.Message):
    """resolution clause, i.e., a clause with id and possibly ids of premises used
    to resolve the clause
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    LITERALS_FIELD_NUMBER: builtins.int
    PREMISES_FIELD_NUMBER: builtins.int
    id: builtins.int
    @property
    def literals(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def premises(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        id: builtins.int = ...,
        literals: collections.abc.Iterable[builtins.int] | None = ...,
        premises: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["id", b"id", "literals", b"literals", "premises", b"premises"]) -> None: ...

global___ResClause = ResClause

class CNFFormula(google.protobuf.message.Message):
    """propositional formula in conjunctive normal form"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NUM_VARS_FIELD_NUMBER: builtins.int
    NUM_CLAUSES_FIELD_NUMBER: builtins.int
    CLAUSES_FIELD_NUMBER: builtins.int
    COMMENTS_FIELD_NUMBER: builtins.int
    SYMBOL_TABLE_FIELD_NUMBER: builtins.int
    num_vars: builtins.int
    num_clauses: builtins.int
    @property
    def clauses(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Clause]: ...
    @property
    def comments(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    @property
    def symbol_table(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    def __init__(
        self,
        *,
        num_vars: builtins.int = ...,
        num_clauses: builtins.int = ...,
        clauses: collections.abc.Iterable[global___Clause] | None = ...,
        comments: collections.abc.Iterable[builtins.str] | None = ...,
        symbol_table: collections.abc.Iterable[builtins.str] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["clauses", b"clauses", "comments", b"comments", "num_clauses", b"num_clauses", "num_vars", b"num_vars", "symbol_table", b"symbol_table"]) -> None: ...

global___CNFFormula = CNFFormula

class CNFSatProblem(google.protobuf.message.Message):
    """propositional satisfiability problem in conjunctive normal form"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FORMULA_FIELD_NUMBER: builtins.int
    TIMEOUT_FIELD_NUMBER: builtins.int
    @property
    def formula(self) -> global___CNFFormula: ...
    timeout: builtins.float
    def __init__(
        self,
        *,
        formula: global___CNFFormula | None = ...,
        timeout: builtins.float = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["formula", b"formula"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["formula", b"formula", "timeout", b"timeout"]) -> None: ...

global___CNFSatProblem = CNFSatProblem

class CNFSatSolution(google.protobuf.message.Message):
    """solution to a propositional satisfiability problem in conjunctive normal form"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STATUS_FIELD_NUMBER: builtins.int
    ASSIGNMENT_FIELD_NUMBER: builtins.int
    CLAUSAL_PROOF_FIELD_NUMBER: builtins.int
    RES_PROOF_FIELD_NUMBER: builtins.int
    status: builtins.str
    @property
    def assignment(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def clausal_proof(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Clause]: ...
    @property
    def res_proof(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ResClause]: ...
    def __init__(
        self,
        *,
        status: builtins.str = ...,
        assignment: collections.abc.Iterable[builtins.int] | None = ...,
        clausal_proof: collections.abc.Iterable[global___Clause] | None = ...,
        res_proof: collections.abc.Iterable[global___ResClause] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["assignment", b"assignment", "clausal_proof", b"clausal_proof", "res_proof", b"res_proof", "status", b"status"]) -> None: ...

global___CNFSatSolution = CNFSatSolution

class ResProofCheckProblem(google.protobuf.message.Message):
    """resolution proof checking problem
    in contrast to the clausal proof checking problem the cnf sat problem is part
    of the proof
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PROOF_FIELD_NUMBER: builtins.int
    TIMEOUT_FIELD_NUMBER: builtins.int
    @property
    def proof(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ResClause]: ...
    timeout: builtins.float
    def __init__(
        self,
        *,
        proof: collections.abc.Iterable[global___ResClause] | None = ...,
        timeout: builtins.float = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["proof", b"proof", "timeout", b"timeout"]) -> None: ...

global___ResProofCheckProblem = ResProofCheckProblem

class ResProofCheckSolution(google.protobuf.message.Message):
    """solution to a resolution proof checking problem"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STATUS_FIELD_NUMBER: builtins.int
    status: builtins.str
    def __init__(
        self,
        *,
        status: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["status", b"status"]) -> None: ...

global___ResProofCheckSolution = ResProofCheckSolution

class ClausalProofCheckProblem(google.protobuf.message.Message):
    """clausal proof checking problem"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FORMULA_FIELD_NUMBER: builtins.int
    PROOF_FIELD_NUMBER: builtins.int
    TIMEOUT_FIELD_NUMBER: builtins.int
    @property
    def formula(self) -> global___CNFFormula: ...
    @property
    def proof(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Clause]: ...
    timeout: builtins.float
    def __init__(
        self,
        *,
        formula: global___CNFFormula | None = ...,
        proof: collections.abc.Iterable[global___Clause] | None = ...,
        timeout: builtins.float = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["formula", b"formula"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["formula", b"formula", "proof", b"proof", "timeout", b"timeout"]) -> None: ...

global___ClausalProofCheckProblem = ClausalProofCheckProblem

class ResProof(google.protobuf.message.Message):
    """resolution proof"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PROOF_FIELD_NUMBER: builtins.int
    @property
    def proof(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ResClause]: ...
    def __init__(
        self,
        *,
        proof: collections.abc.Iterable[global___ResClause] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["proof", b"proof"]) -> None: ...

global___ResProof = ResProof