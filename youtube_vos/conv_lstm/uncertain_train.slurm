#!/bin/bash

#SBATCH --output=us_try_3_seg_const_l2_uncertain_3_frames_10_90_0.1.out
#SBATCH --error=e1.err
#SBATCH --job-name=ussl13

#SBATCH --nodes=1
#SBATCH --time=45:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu32
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

source activate /home/akumar/my-envs/ak1

echo 'Beginning script'

python unsup_sup_try_3_seg_const_uncertainty_ramp.py --batch_size 4 --epochs 50 --pf 100 --wt_cons 0.1 --exp_id us_try_3_seg_const_l2_uncertain_3_frames_10_90_0.1

echo "Ending script..."

