"""ML2 global variables"""

import os

# local storage directory

LOCAL_STORAGE_DIR = os.path.expanduser(os.environ.get("ML2_LOCAL_STORAGE_DIR", "~/ml2-storage"))

# Google Cloud Platform

ML2_BUCKET = os.environ.get("ML2_GCP_BUCKET", "ml2-public")

# Docker

CONTAINER_REGISTRY = os.environ.get("ML2_CONTAINER_REGISTRY", "ghcr.io/reactive-systems/ml2")

# Weights and Biases

WANDB_ENTITY = os.environ.get("ML2_WANDB_ENTITY")

# Propositional satisfiability

PROP_SAT_ALIASES = {}
PROP_SAT_PROJECT_NAME = "prop-sat"

# LTL satisfiability

LTL_SAT_ALIASES = {}
LTL_SAT_PROJECT_NAME = "ltl-sat"

# LTL specifications

LTL_SPEC_ALIASES = {"sc20": "sc-0", "scp-ni5-no5": "scp-0", "scp-ni5-no5-ts25": "scp-1"}
LTL_SPEC_PROJECT_NAME = "ltl-spec"

# LTL synthesis

LTL_SYN_ALIASES = {}
LTL_SYN_PROJECT_NAME = "ltl-syn"
