#!/bin/bash
#SBATCH -J testOthers
#SBATCH -p koni
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --nodelist=gpu47
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --comment python

export MASTER_ADDR=$(hostname --ip)
export MASTER_PORT=$((10#${SLURM_JOB_ID: -4} + 20002))
export PYTORCH_ALLOC_CONF=expandable_segments:True


ml singularity/4.1.0 

# hyperclovax-omni
singularity run --nv container/torch290 \
python test_others_hf.py
