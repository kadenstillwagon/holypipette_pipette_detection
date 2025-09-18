#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=10
#SBATCH --error=/storage/coda1/p-cf76/0/kstillwagon6/SAMCell_Kaden_PACE/dinov2/dinov2/run/checkpoint/outputs/%j_0_log.err
#SBATCH --gpus-per-node=8
#SBATCH --job-name=dinov2:train
#SBATCH --mem=0GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/storage/coda1/p-cf76/0/kstillwagon6/SAMCell_Kaden_PACE/dinov2/dinov2/run/checkpoint/outputs/%j_0_log.out
#SBATCH --partition=learnlab
#SBATCH --signal=USR2@120
#SBATCH --time=2800
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /storage/coda1/p-cf76/0/kstillwagon6/SAMCell_Kaden_PACE/dinov2/dinov2/run/checkpoint/outputs/%j_%t_log.out --error /storage/coda1/p-cf76/0/kstillwagon6/SAMCell_Kaden_PACE/dinov2/dinov2/run/checkpoint/outputs/%j_%t_log.err /usr/local/pace-apps/manual/packages/anaconda3/2023.03/bin/python3 -u -m submitit.core._submit /storage/coda1/p-cf76/0/kstillwagon6/SAMCell_Kaden_PACE/dinov2/dinov2/run/checkpoint/outputs
