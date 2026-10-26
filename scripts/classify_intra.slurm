#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=28:00:00
#SBATCH --output=neuropath2path-intra-%j.log
#SBATCH --job-name=neuropath2path-intra

set -euo pipefail
# Run sbatch from the repository root; activate your Python environment first.
# Add your cluster's partition/account directives as required.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m neuropath2path classify "${1:-Intracellular_data.zip}" \
  --mode intra --test-proportion 0.1 --bin-width 7 --search-step 5 \
  --neighbors 5 --search-span 10 --seed 100 \
  --jobs "${SLURM_CPUS_PER_TASK:-1}" --output results/intra.json
