#!/bin/bash
#SBATCH --job-name=ink_labels_synth_training
#SBATCH --output=Log.log
#SBATCH --error=Error.err
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=25:00:00
#SBATCH --cpus-per-task=10
#SBATCH --partition=main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16

source ~/ies_env/bin/activate
cd /mnt/stud/home/npopkov/grandchallenges_vesuvius/notebooks
srun python training_script_slurm.py
