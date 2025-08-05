#!/bin/bash

# This script generates the HGEMM kernel for the given parameters.

# Test cases
cases=(
  "Case1 768 768 768"
  "Case2 128 1024 2048"
  "Case3 128 2048 8192"
  "Case4 512 3072 1024"
  "Case5 512 4096 8192"
  "Case6 3136 576 64"
  "Case7 4096 4096 4096"
  "Case8 1024 16384 16384"
  "Case9 4096 16384 14336"
  "Case10 32768 32768 32768"
)

# Output directory
output_dir="data/input"
mkdir -p "$output_dir"

for case in "${cases[@]}"; do
  read -r name M N K <<<"$case"

  case_dir="${output_dir}/${name}_${M}x${N}x${K}"
  mkdir -p "$case_dir"

  echo "Generating matrices of ${name} (M=$M, N=$N, K=$K)"

  python3 tools/matrix_generation.py --M "$M" --N "$N" --K "$K" --outdir "$case_dir"

  if [ $? -ne 0 ]; then
    echo "Failed to generate binary matrices for $name"
    exit 1
  fi
done

echo "All binary matrix generated successfully."