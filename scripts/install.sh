#!/bin/bash
uv venv .venv
source .venv/bin/activate

uv sync
uv pip install -e .

# training
uv pip install timm
uv pip install wandb
uv pip install datasets
uv pip install deepspeed
uv pip install accelerate
uv pip install setuptools
uv pip install --upgrade tokenizers
uv pip install flash-attn --no-build-isolation

uv pip install transformers
uv pip install lm-eval["longbench"]

# interactive tools
uv pip install ipython gpustat
#