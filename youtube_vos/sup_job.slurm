#!/bin/bash

#SBATCH --output=sup_20_per_labeled.out
#SBATCH --error=e1.err
#SBATCH --job-name=as1

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu16
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

source activate /home/akumar/my-envs/ak1

echo 'Beginning script'

python percent_label_train.py --batch_size 4 --epochs 25 --exp_id sup_20_per_labeled

echo "Ending script..."

