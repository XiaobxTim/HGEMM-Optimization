#!/bin/bash

# program
BIN1=/work/sustcsc_12/xbx/HGEMM/build/hgemm_custom_large
BIN2=/work/sustcsc_12/xbx/HGEMM/build/hgemm_custom

INPUT_ROOT="data/input"
OUTPUT_ROOT="data/output/custom_results"

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

mkdir -p "$OUTPUT_ROOT"

for case in "${cases[@]}"; do
  read -r name M N K <<< "$case"

  input_dir="${INPUT_ROOT}/${name}_${M}x${N}x${K}"
  output_file="${OUTPUT_ROOT}/custom_result_${name}_${M}x${N}x${K}.txt"

  if [ ! -d "$input_dir" ]; then
    echo "❌ Input directory $input_dir not found, skipping $name ..."
    continue
  fi

  echo "🚀 Running $name with M=$M N=$N K=$K"

  # 根据案例选择不同的二进制程序
  if [[ "$name" == "Case8" || "$name" == "Case9" || "$name" == "Case7" ]]; then
    BIN="$BIN1"
  else
    BIN="$BIN2"
  fi

  $BIN --indir "$input_dir" --outdir "$OUTPUT_ROOT"

  if [ $? -ne 0 ]; then
    echo "❌ Run failed for $name"
    exit 1
  fi

  echo "✅ Finished $name, output saved to custom_result_${name}_${M}x${N}x${K}.txt"
  echo ""
done

echo "🎉 All benchmark cases processed."