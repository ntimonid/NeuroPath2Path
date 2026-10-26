#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=28:00:00
#SBATCH --output=neuropath2path-weights-%j.log
#SBATCH --job-name=neuropath2path-weights

set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m neuropath2path weights "${1:-Intercellular_data.zip}" \
  --seed 100 --output results/weights.json
