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
#SBATCH --array=0-2015


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

# Define all argument combinations into an array for parallel job submission
all_args=()
for fn in "NA_ALISTA_UR_128" "ALISTA_AT" "ALISTA" "FISTA" "ISTA" "AGLISTA" "NA_ALISTA_U_128" "NA_ALISTA_R_128"; do
  for k in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 ; do
      for n in 500 750 1000 1250 1500 1750 2000 2250 2500 ; do
          for snr in 40 20 ; do
              all_args+=("-k $k -n $n -N $snr -f $fn")
          done
      done
    done
done

args=${all_args[$SLURM_ARRAY_TASK_ID]}

echo "Executing job $((SLURM_ARRAY_TASK_ID+1))/${#all_args[@]} with the following arguments:"
echo $args
echo ""

singularity run --nv --bind "${CODE_DIR}:${CODE_MNT}" ./slurm-gpu/torch170-cuda110.sif run.py $args
