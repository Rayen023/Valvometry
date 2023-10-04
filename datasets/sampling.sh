#!/bin/bash
#SBATCH --account=def-selouani
#SBATCH --mem=64G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-04:00
#SBATCH --output=output/%N-%j.out

module load python/3.9 scipy-stack

source ../ENV/bin/activate


python plots.py
