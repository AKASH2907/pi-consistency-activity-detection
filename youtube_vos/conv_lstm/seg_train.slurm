#!/bin/bash

#SBATCH --output=us_try_1_seg_const_l2_10_90.out
#SBATCH --error=e1.err
#SBATCH --job-name=ussl11mp

#SBATCH --nodes=1
#SBATCH --time=50:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu32
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

source activate /home/akumar/my-envs/ak1

echo 'Beginning script'

python unsup_sup_try_1_seg_const.py --batch_size 4 --epochs 30 --pf 100 --wt_cons 1 --exp_id us_try_1_seg_const_l2_10_90

echo "Ending script..."

