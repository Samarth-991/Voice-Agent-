# Sourced by the setup/run scripts. /mnt is the compute's large, fast, but
# EPHEMERAL disk (wiped on stop/deallocate) — rerun scripts/setup_env.sh after.
export DRA_CACHE="${DRA_CACHE:-/mnt/cache/deepagent-rtv}"
export DRA_VENV="${DRA_VENV:-$DRA_CACHE/venv}"
# Base interpreter for the venv (deepagents needs Python >= 3.11).
export DRA_BASE_PYTHON="${DRA_BASE_PYTHON:-/anaconda/envs/pyagents/bin/python}"

export HF_HOME="$DRA_CACHE/hf"
export TORCH_HOME="$DRA_CACHE/torch"
export PIP_CACHE_DIR="$DRA_CACHE/pip"
export TMPDIR="${TMPDIR:-/mnt/tmp}"
export HTTP_PROXY=""
export HTTPS_PROXY=""