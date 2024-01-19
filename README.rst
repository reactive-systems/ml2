================================================
ML2: Machine Learning for Mathematics and Logics
================================================


.. image:: https://img.shields.io/pypi/pyversions/ml2
    :target: https://www.python.org
.. image:: https://img.shields.io/pypi/v/ml2
    :target: https://pypi.org/project/ml2/
.. image:: https://img.shields.io/github/license/reactive-systems/ml2 
    :target: https://github.com/reactive-systems/ml2/blob/main/LICENSE


ML2 is an open source Python library for machine learning research on mathematical and logical reasoning problems. The library includes the (re-)implementation of the research papers `Teaching Temporal Logics to Neural Networks <https://iclr.cc/virtual/2021/poster/3332>`_ and `Neural Circuit Synthesis from Specification Patterns <https://proceedings.neurips.cc/paper/2021/file/8230bea7d54bcdf99cdfe85cb07313d5-Paper.pdf>`_. So far, the focus of ML2 is on propositional and linear-time temporal logic (LTL) and transformer architectures. ML2 is actively developed at `CISPA Helmholtz Center for Information Security <https://cispa.de/en>`_.


Requirements
------------

- `Docker <https://www.docker.com>`_
- `Python 3.8 <https://www.python.org/dev/peps/pep-0569/>`_

Note on Docker: For data generation, evaluation, and benchmarking ML2 uses a variety of research tools (e.g. SAT solvers, model checkers, and synthesis tools). For ease of use, each tool is encapsulated in a Docker container that is automatically pulled and started when the tool is needed. Thus, ML2 requires Docker to be installed and running.

Installation
------------

**Before installing ML2, please note the Docker requirement.**

From PyPI
~~~~~~~~~

Install ML2 from PyPI with ``pip install ml2``.

From Source
~~~~~~~~~~~

To install ML2 from source, clone the git repo and install with pip as follows:

.. code:: shell

    git clone https://github.com/reactive-systems/ml2.git && \
    cd ml2 && \
    pip install .

For development pip install in editable mode and include the development dependencies as follows:

.. code:: shell

    pip install -e .[dev]


Neural Circuit Synthesis (`presented at NeurIPS 21 <https://proceedings.neurips.cc/paper/2021/file/8230bea7d54bcdf99cdfe85cb07313d5-Paper.pdf>`_)
--------------------------------------------------------------------------------------------------------------------------------------------------------

In this project, hierarchical Transformers were trained to synthesize hardware circuits directly out of high-level specifications in a temporal logic. The lack of sufficient amounts of training data was tackled with a method to generate large amounts of additional training data, i.e., pairs of specifications and circuits implementing them by mining common specification patterns from the annual synthesis competition `SYNTCOMP <syntcomp.org>`_.

Training
~~~~~~~~

To train a hierarchical Transformer with default parameters:

.. code:: shell

    python -m ml2.experiment.run configs/ltl-syn/neurips21/ht.json

Evaluation
~~~~~~~~~~

To evaluate a hierarchical Transformer trained on the circuit synthesis task:

.. code:: shell

    python -m ml2.experiment.run configs/ltl-syn/neurips21/ht-sem-eval.json`

Datasets and Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate a dataset of specifications and AIGER circuits:

.. code:: shell
    
    python -m ml2.ltl.ltl_syn.ltl_syn_data_gen_patterns --name dataset`

How to Cite
~~~~~~~~~~~

.. code:: tex

    @inproceedings{neural_circuit_synthesis,
        author    = {Frederik Schmitt and Christopher Hahn and Markus N. Rabe and Bernd Finkbeiner},
        title     = {Neural Circuit Synthesis from Specification Patterns},
        booktitle = {Advances in Neural Information Processing Systems 34 Pre-proceedings},
        year      = {2021}
        url       = {https://proceedings.neurips.cc/paper/2021/hash/8230bea7d54bcdf99cdfe85cb07313d5-Abstract.html}
    }


Teaching Temporal Logics to Neural Networks (`presented at ICLR 21 <https://iclr.cc/virtual/2021/poster/3332>`_)
-------------------------------------------------------------------------------------------------------------------

In this project, Transformers were trained on the problem of finding a satisfying trace to a linear-time temporal logic (LTL) formula. While the training data was generated with classical solvers, providing only one of a possibly infinite number of solutions, the Transformers successfully generalized: while often deviating from the solutions found by the classical solver, they still predicted a correct solution to most formulas. Generalization was also demonstrated on larger formulas and formulas on which the classical solver timed out.

Training
~~~~~~~~

To train a Transformer with default parameters on the trace generation problem:

.. code:: shell

    python -m ml2.experiment.run configs/ltl-strace/iclr21/t.json

For the propositional satisfiability experiment:

.. code:: shell

    python -m ml2.experiment.run configs/prop-sat/iclr21/t.json

Evaluation
~~~~~~~~~~

To evaluate a Transformer trained on the trace generation problem:

.. code:: shell

    python -m ml2.experiment.run configs/ltl-strace/iclr21/t-sem-eval.json`

How to Cite
~~~~~~~~~~~

.. code:: tex

    @inproceedings{teaching_temporal_logics,
        title     = {Teaching Temporal Logics to Neural Networks},
        author    = {Christopher Hahn and Frederik Schmitt and Jens U. Kreber and Markus N. Rabe and Bernd Finkbeiner},
        booktitle = {International Conference on Learning Representations},
        year      = {2021},
        url       = {https://openreview.net/forum?id=dOcQK-f4byz}
    }
