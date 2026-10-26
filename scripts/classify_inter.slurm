#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=28:00:00
#SBATCH --output=neuropath2path-inter-%j.log
#SBATCH --job-name=neuropath2path-inter

set -euo pipefail
# Run sbatch from the repository root; activate your Python environment first.
# Add your cluster's partition/account directives as required.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m neuropath2path classify "${1:-Intercellular_data.zip}" \
  --mode inter --test-proportion 0.1 --bin-width 15 --search-step 5 \
  --neighbors 5 --search-span 9 --seed 100 \
  --jobs "${SLURM_CPUS_PER_TASK:-1}" --output results/inter.json
