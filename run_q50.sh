#!/bin/bash -l
#SBATCH --job-name=q50-qxf
#SBATCH --account=project_462001763
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=q50-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module use /appl/local/quantum/modulefiles
module use /appl/local/quantum/modulefiles/hackathon
module load fiqci-vtt-qiskit-QxF/v1.0

if [ -z "${Q50_CORTEX_URL:-}" ]; then
  echo "Q50_CORTEX_URL is empty"
  exit 1
fi
echo "Q50 endpoint is set"
python -u hardware_batch.py --backend q50
