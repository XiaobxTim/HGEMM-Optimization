# Compiler--NVCC
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -arch=sm_70 -Xcompiler "-fopenmp -O3 -use_fast_math" 

# Source files
SRC_DIR = src

# Object files
BLAS_SRC = $(SRC_DIR)/hgemm_cublas.cu
CUSTOM_SRC = $(SRC_DIR)/hgemm.cu
CUSTOM_LARGE_SRC = $(SRC_DIR)/hgemm_large.cu
# BENCH_SRC = $(SRC_DIR)/hgemm_cublas_bench.cu
COMPARE_SRC = $(SRC_DIR)/hgemm_compare.cu
COMPARE_LARGE_SRC = $(SRC_DIR)/hgemm_compare_large.cu

TARGET_CUBLAS = hgemm_cublas
TARGET_CUSTOM = hgemm_custom
TARGET_CUSTOM_LARGE = hgemm_custom_large
# TARGET_CUBLAS_BENCH = hgemm_cublas_bench
TARGET_COMPARE = hgemm_compare
TARGET_COMPARE_LARGE = hgemm_compare_large

# Default target
# all: $(TARGET_CUBLAS) $(TARGET_CUSTOM) $(TARGET_CUBLAS_BENCH) $(TARGET_COMPARE)
all: $(TARGET_CUBLAS) $(TARGET_CUSTOM) $(TARGET_CUSTOM_LARGE) $(TARGET_COMPARE) $(TARGET_COMPARE_LARGE)

# Build rules
build:
	mkdir -p build

$(TARGET_CUBLAS): build $(BLAS_SRC)
	$(NVCC) $(NVCC_FLAGS) $(BLAS_SRC) -o build/$(TARGET_CUBLAS) -lcublas -lcudart

# $(TARGET_CUBLAS_BENCH): build $(BENCH_SRC)
# 	$(NVCC) $(NVCC_FLAGS) $(BENCH_SRC) -o build/$(TARGET_CUBLAS_BENCH) -lcublas -lcudart

$(TARGET_COMPARE): build $(COMPARE_SRC)
	$(NVCC) $(NVCC_FLAGS) $(COMPARE_SRC) -o build/$(TARGET_COMPARE) -lcublas -lcudart

$(TARGET_CUSTOM): build $(CUSTOM_SRC)
	$(NVCC) $(NVCC_FLAGS) $(CUSTOM_SRC) -o build/$(TARGET_CUSTOM) -lcudart

$(TARGET_COMPARE_LARGE): build $(COMPARE_LARGE_SRC)
	$(NVCC) $(NVCC_FLAGS) $(COMPARE_LARGE_SRC) -o build/$(TARGET_COMPARE_LARGE) -lcublas -lcudart

$(TARGET_CUSTOM_LARGE): build $(CUSTOM_LARGE_SRC)
	$(NVCC) $(NVCC_FLAGS) $(CUSTOM_LARGE_SRC) -o build/$(TARGET_CUSTOM_LARGE) -lcudart

# Clean rule
clean:
	rm -f build/*

# Phony targets
.PHONY: all build clean