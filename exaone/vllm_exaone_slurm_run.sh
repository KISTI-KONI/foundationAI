#!/bin/bash
#SBATCH -J testExaone
#SBATCH -p koni
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --comment python

export MASTER_ADDR=$(hostname --ip)
export MASTER_PORT=$((10#${SLURM_JOB_ID: -4} + 20002))
export PYTORCH_ALLOC_CONF=expandable_segments:True

ml singularity/4.1.0 

singularity exec --nv container/torch290.sif bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate /home01/r919a12/.conda/envs/exaone_vllm
vllm serve LGAI-EXAONE/K-EXAONE-236B-A23B \
  --reasoning-parser deepseek_v3 \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
"

