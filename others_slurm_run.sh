#!/bin/bash
#SBATCH -J testOthers
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

# 공통 프롬프트
export PROMPT="Which one is bigger, 3.9 vs 3.12?"

ml singularity/4.1.0 

# HyperCLOVAX - 싱글 노드 동작 확인
singularity run --nv container/torch290 \
python test_others_hf.py \
    --model_id naver-hyperclovax/HyperCLOVAX-SEED-Think-32B \
    --prompt "$PROMPT" \
    --output_path results/hyperclovax_qa.json

# Solar - 싱글 노드 동작 확인
singularity run --nv container/torch290 \
python test_others_hf.py \
    --model_id upstage/Solar-Open-100B \
    --prompt "$PROMPT" \
    --output_path results/solar_qa.json

# VAETKI - 싱글 노드 동작 확인 
singularity run --nv container/torch290 \
python test_others_hf.py \
    --model_id NC-AI-consortium-VAETKI/VAETKI \
    --prompt "$PROMPT" \
    --output_path results/vaetki_qa.json

# AX (허깅페이스 분산) - 싱글 노드 동작 확인
singularity run --nv container/torch290 \
python test_others_hf.py \
    --model_id skt/A.X-K1 \
    --prompt "$PROMPT" \
    --output_path results/sk_axk1_qa.json



