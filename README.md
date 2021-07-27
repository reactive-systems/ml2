# <span style="color:Maroon">ML</span><span style="color:Orange">2</span>: <span style="color:Maroon">M</span>achine <span style="color:Maroon">L</span>earning for <span style="color:Orange">M</span>athematics and <span style="color:Orange">L</span>ogics

ML2 is an open source library for Machine Learning for Mathematics and Logics. The implementation, so far, focuses on sequence to sequence models, such as the [Transformer](https://arxiv.org/abs/1706.03762) and [hierarchical Transformer](https://arxiv.org/abs/2006.09265).

ML2 was developed at CISPA Helmholtz Center for Information Security to provide a flexible codebase for Machine Learning research on mathematical and logical reasoning problems.
The library is based on the code developed for [DeepLTL](https://github.com/reactive-systems/deepltl) the following research paper.

- [Neural Circuit Synthesis from Specification Patterns (arXiv'21)](https://arxiv.org/abs/2107.11864): A hierarchical Transformer was trained to synthesize hardware circuits directly out of high-level specifications in a temporal logic. The lack of sufficient amounts of training data was tackled a method to generate large amounts of additional training data, i.e., pairs of specifications and circuits implementing them by mining common specification patterns from the annual synthesis competition ([SYNTCOMP](syntcomp.org)).

## Requirements

- [Docker](https://www.docker.com/)
- [Python 3.7](https://www.python.org/downloads/release/python-370/)

## Installation

Install ML2 with pip in the project directory containing the [setup file](setup.cfg) as follows

`pip install .`

To install ML2 in editable mode and include the development dependencies run in the project directory containing the [setup file](setup.cfg) the following command

`pip install -e .[dev]`

## Neural Circuit Synthesis

### Training

To train a hierarchical Transformer with default parameters run

`python -m ml2.ltl.ltl_syn.ltl_syn_hier_transformer_experiment train`

### Evaluation

To evaluate the hierarchical Transformer from our paper run

`python -m ml2.ltl.ltl_syn.ltl_syn_hier_transformer_experiment eval -n hier-transformer-0`

### Datasets and Data Generation

To generate a dataset of specifications and AIGER circuits run

`python -m ml2.ltl.ltl_syn.ltl_syn_data_gen_patterns --name dataset`

### How to Cite

```
@article{neuralCircuitSynt,
author = {Frederik Schmitt and Christopher Hahn and Markus N. Rabe and Bernd Finkbeiner},
title = {Neural Circuit Synthesis from Specification Patterns},
journal = {arXiv},
year = {2021}
}
```
