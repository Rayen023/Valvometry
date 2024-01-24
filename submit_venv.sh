#!/bin/bash
#SBATCH --account=def-selouani
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --mem=96G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-24:00
#SBATCH --output=output/%N-%j.out

echo "Loading Python 3.9 and scipy-stack module..."
module load python/3.9 scipy-stack

echo "Activating virtual environment..."
source ENV/bin/activate

echo "Running script.py..."
python optuna_cv_train.py
