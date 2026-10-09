#!/bin/bash
#SBATCH --job-name=aoi-marl
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=aoi-%j.out

source ~/aoi-venv/bin/activate

python -u MARL_RA.py