#!/bin/bash

#SBATCH --output=sup_100per_labeled_mp_ep100.out
#SBATCH --error=e1.err
#SBATCH --job-name=ssl1mp

#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu32
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

source activate /home/akumar/my-envs/ak1

echo 'Beginning script'

python percent_label_train_hp.py --batch_size 16 --epochs 100 --exp_id sup_100per_labeled_mp_ep100 --pf 50

echo "Ending script..."

