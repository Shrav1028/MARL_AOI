#!/bin/bash
#SBATCH --job-name=aoi-marl
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=aoi-%j.out

source ~/aoi-venv/bin/activate

python -u MARL_RA.py