#!/bin/bash
#SBATCH --account=def-selouani
#SBATCH --gres=gpu:v100:1       # Request GPU "generic resources"
#SBATCH --mem=96G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-24:00
#SBATCH --output=output/%N-%j.out

module load python/3.9 scipy-stack

source ENV/bin/activate

python training_optuna.py
