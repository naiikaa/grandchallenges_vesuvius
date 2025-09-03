#!/bin/bash
#SBATCH --output=Log.log
#SBATCH --error=Error.err
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=25:00:00
#SBATCH --cpus-per-task=10
#SBATCH --partition=main
#SBATCH --gres=gpu:4

# get arguments
sample_size=$1
volume_depth=$2


source ~/vesuvius/bin/activate
cd /mnt/stud/home/npopkov/grandchallenges_vesuvius/src
srun python AAE_train.py --sample_size $1 --volume_depth $2