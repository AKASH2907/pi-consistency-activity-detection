#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=7
#SBATCH -C gmem12
#SBATCH --mem-per-cpu=5G
#SBATCH -p gpu
module load anaconda3
echo "Start inference reuVOS"
python inference.py
