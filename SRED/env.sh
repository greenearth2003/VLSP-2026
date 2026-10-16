# source env.sh before running anything
_SRED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
export TORCH_HOME=${_SRED_ROOT}/.cache/torch
export PYTHONWARNINGS=ignore
export SRED_DATA=${SRED_DATA:-${_SRED_ROOT}/data/Vi-SRED26}
