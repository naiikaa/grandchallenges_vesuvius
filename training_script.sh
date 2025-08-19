#!/bin/bash
#SBATCH --job-name="condDDPM vesuvius experiment"
#SBATCH --output=Log.log
#SBATCH --error=Error.err
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=25:00:00
#SBATCH --cpus-per-task=10
#SBATCH --partition=main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16

# get arguments
sample_size=$1
volume_depth=$2


source ~/ies_env/bin/activate
cd /mnt/stud/home/npopkov/grandchallenges_vesuvius/src
srun python AAE_train.py --sample_size sample_size --volume_depth volume_depth