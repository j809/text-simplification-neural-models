#!/bin/bash
#
#SBATCH --job-name=run.sh
#SBATCH --output=train_log.txt  # output file
#SBATCH -e log.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to 
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=4:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4000    # Memory in MB per cpu allocated
hostname
sleep 1
exit
