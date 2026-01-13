#!/usr/bin/env bash
set -euo pipefail

ml singularity/4.1.0

# 실행 위치 기준으로 container 디렉토리로 이동
cd container

# 1) Docker 이미지로부터 SIF 빌드
singularity build --fakeroot torch290.sif docker://pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel

# 2) 로컬 def 파일로 sandbox 빌드
singularity build --fakeroot --sandbox ./exaone/ torch290_git_local.def



