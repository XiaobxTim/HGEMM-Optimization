#!/bin/bash
#SBATCH --job-name=hgemm
#SBATCH --partition=8v100-32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=hgemm_%j.out
#SBATCH --error=hgemm_%j.err

source /work/sustcsc_12/miniconda3/etc/profile.d/conda.sh
conda activate xbx

cd /work/sustcsc_12/xbx/HGEMM

# Set CUDA environment
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Clean and Compile the project
make clean
make

# bash ./scripts/compare.sh
# bash ./scripts/custom_benchmark.sh
nsys profile --stats=true --trace=cuda,nvtx --cuda-memory-usage=true ./build/hgemm_custom --indir /work/sustcsc_12/xbx/HGEMM/data/input/Case7_4096x4096x4096 -outdir /work/sustcsc_12/xbx/HGEMM/data/output/profile
# bash ./scripts/custom_profile.sh