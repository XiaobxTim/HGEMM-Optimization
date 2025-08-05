#!/bin/bash

# Test script for HGEMM with Coppersmith-Winograd optimizations
# This script compiles and tests different HGEMM implementations

set -e  # Exit on any error

echo "=== HGEMM Coppersmith-Winograd Optimization Test ==="
echo "Date: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Check if CUDA is available
check_cuda() {
    print_status "Checking CUDA availability..."
    if command -v nvcc &> /dev/null; then
        print_status "NVCC found: $(nvcc --version | head -n1)"
    else
        print_error "NVCC not found. Please install CUDA toolkit."
        exit 1
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        print_status "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        print_warning "nvidia-smi not found. GPU information may be limited."
    fi
    echo ""
}

# Compile all versions
compile_versions() {
    print_header "Compiling HGEMM versions..."
    
    # Clean previous builds
    make clean
    
    # Compile all versions
    print_status "Compiling original HGEMM..."
    make hgemm_original
    
    print_status "Compiling CW-optimized HGEMM..."
    make hgemm_cw_optimized
    
    print_status "Compiling CW-advanced HGEMM..."
    make hgemm_cw_advanced
    
    print_status "Compilation completed successfully!"
    echo ""
}

# Check if test data exists
check_test_data() {
    print_status "Checking test data availability..."
    
    if [ ! -d "data/input/Case1_768x768x768" ]; then
        print_error "Test data not found. Please ensure data/input/Case1_768x768x768 exists."
        exit 1
    fi
    
    if [ ! -f "data/input/Case1_768x768x768/A_matrix.bin" ] || [ ! -f "data/input/Case1_768x768x768/B_matrix.bin" ]; then
        print_error "Matrix files not found in test data directory."
        exit 1
    fi
    
    print_status "Test data found and verified."
    echo ""
}

# Run performance tests
run_performance_tests() {
    print_header "Running performance tests..."
    
    # Create output directory
    mkdir -p data/output
    
    # Test original HGEMM
    print_status "Testing original HGEMM..."
    ./hgemm_original -d data/input/Case1_768x768x768 -o data/output
    
    # Test CW-optimized HGEMM
    print_status "Testing CW-optimized HGEMM..."
    ./hgemm_cw_optimized -d data/input/Case1_768x768x768 -o data/output
    
    # Test CW-advanced HGEMM
    print_status "Testing CW-advanced HGEMM..."
    ./hgemm_cw_advanced -d data/input/Case1_768x768x768 -o data/output
    
    print_status "Performance tests completed!"
    echo ""
}

# Generate performance report
generate_report() {
    print_header "Generating performance report..."
    
    # Check if Python is available
    if command -v python3 &> /dev/null; then
        print_status "Running performance analysis..."
        python3 performance_analysis.py
    else
        print_warning "Python3 not found. Skipping detailed analysis."
        print_status "Manual comparison of results:"
        echo ""
        
        # Show basic comparison
        if [ -f "data/output/result_Case1_768x768x768.txt" ]; then
            echo "=== Original HGEMM Results ==="
            cat data/output/result_Case1_768x768x768.txt
            echo ""
        fi
        
        if [ -f "data/output/result_cw_Case1_768x768x768.txt" ]; then
            echo "=== CW-Optimized HGEMM Results ==="
            cat data/output/result_cw_Case1_768x768x768.txt
            echo ""
        fi
        
        if [ -f "data/output/result_cw_advanced_Case1_768x768x768.txt" ]; then
            echo "=== CW-Advanced HGEMM Results ==="
            cat data/output/result_cw_advanced_Case1_768x768x768.txt
            echo ""
        fi
    fi
}

# Quick validation test
quick_validation() {
    print_header "Running quick validation test..."
    
    # Test with a small matrix to ensure correctness
    print_status "Testing basic functionality..."
    
    # This would require a small test case
    # For now, just check if executables run without errors
    if ./hgemm_original -d data/input/Case1_768x768x768 -o data/output > /dev/null 2>&1; then
        print_status "Original HGEMM: ✓ Basic functionality OK"
    else
        print_error "Original HGEMM: ✗ Basic functionality failed"
    fi
    
    if ./hgemm_cw_optimized -d data/input/Case1_768x768x768 -o data/output > /dev/null 2>&1; then
        print_status "CW-Optimized HGEMM: ✓ Basic functionality OK"
    else
        print_error "CW-Optimized HGEMM: ✗ Basic functionality failed"
    fi
    
    if ./hgemm_cw_advanced -d data/input/Case1_768x768x768 -o data/output > /dev/null 2>&1; then
        print_status "CW-Advanced HGEMM: ✓ Basic functionality OK"
    else
        print_error "CW-Advanced HGEMM: ✗ Basic functionality failed"
    fi
    
    echo ""
}

# Main execution
main() {
    print_header "Starting HGEMM Coppersmith-Winograd optimization test"
    echo ""
    
    # Check prerequisites
    check_cuda
    check_test_data
    
    # Compile versions
    compile_versions
    
    # Quick validation
    quick_validation
    
    # Run performance tests
    run_performance_tests
    
    # Generate report
    generate_report
    
    print_header "Test completed successfully!"
    echo ""
    print_status "Results are available in data/output/"
    print_status "For detailed analysis, run: python3 performance_analysis.py"
}

# Handle command line arguments
case "${1:-}" in
    "clean")
        print_status "Cleaning build artifacts..."
        make clean
        rm -f *.png *.json
        print_status "Clean completed."
        ;;
    "compile")
        check_cuda
        compile_versions
        ;;
    "test")
        check_test_data
        run_performance_tests
        ;;
    "report")
        generate_report
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  (no args)  Run complete test suite"
        echo "  clean       Clean build artifacts"
        echo "  compile     Compile all versions"
        echo "  test        Run performance tests"
        echo "  report      Generate performance report"
        echo "  help        Show this help message"
        ;;
    *)
        main
        ;;
esac 