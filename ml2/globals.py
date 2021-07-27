"""ML2 global variables"""

import os

# local storage directory

LOCAL_STORAGE_DIR = os.path.expanduser('~/ml2-storage')

# Google Cloud Platform

ML2_BUCKET = os.environ.get('ML2_GCP_BUCKET', 'ml2-bucket')

# Docker
IMAGE_BASE_NAME = os.environ.get('ML2_IMAGE_BASE_NAME',
                                 'ghcr.io/reactive-systems/ml2')

# Weights and Biases

WANDB_ENTITY = os.environ.get('ML2_WANDB_ENTITY')

# LTL specifications

LTL_SPEC_ALIASES = {
    'sc20': 'sc-0',
    'scp-ni5-no5': 'scp-0',
    'scp-ni5-no5-ts25': 'scp-1'
}
LTL_SPEC_BUCKET_DIR = 'ltl-spec'
LTL_SPEC_WANDB_PROJECT = 'ltl-spec'

# LTL synthesis

LTL_SYN_ALIASES = {}
LTL_SYN_BUCKET_DIR = 'ltl-syn'
LTL_SYN_WANDB_PROJECT = 'ltl-syn'