#!/bin/bash
#
### ADAPT TO YOUR PREFERRED SLURM OPTIONS ###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=max.mustermann@hhi.fraunhofer.de
#SBATCH --job-name="lista"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=4GB
##SBATCH --array=0-3


# include the definition of the LOCAL_JOB_DIR which is autoremoved after each job
source "/etc/slurm/local_job_dir.sh"

# Use input as source code directory if present, otherwise resort to default
if [ $# -eq 0 ]
  then
    CODE_DIR="/data/cluster/users/${USER}/lista"
  else
    CODE_DIR=$1
fi
CODE_MNT="/mnt/project"

singularity run --nv --bind ${CODE_DIR}:${CODE_MNT} ./slurm-gpu/torch170-cuda110.sif run.py
