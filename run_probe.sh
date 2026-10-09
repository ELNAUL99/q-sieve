#!/bin/bash -l
#SBATCH --job-name=q50-probe
#SBATCH --account=project_462001763
#SBATCH --partition=small
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=probe-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module use /appl/local/quantum/modulefiles
module use /appl/local/quantum/modulefiles/hackathon
module load fiqci-vtt-qiskit-QxF/v1.0
python -u probe_q50.py
