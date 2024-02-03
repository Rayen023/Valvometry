#!/bin/bash
#SBATCH --account=def-selouani
#SBATCH --mem=270G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-16:00
#SBATCH --output=output/%N-%j.out

module load python/3.10 scipy-stack

echo "Activating virtual environment..."
source ENV/bin/activate

echo "Running script.py..."
python cv_oysters_valvometry_rec.py
