#!/bin/bash

################################################################################
# Comprehensive Baseline Evaluation Runner
#
# Runs all evaluation scripts for the retrieval system:
# 1. evaluate_baselines_with_reranking.py - Individual baseline + LLM reranking
# 2. compare_baselines_vs_system.py - Full system vs baselines
# 3. generate_individual_graphs.py - Generate visualization graphs
#
# Usage:
#   ./run_evaluation.sh [options]
#
# Options:
#   -n NUM      Number of examples to evaluate (default: 50)
#   -k K        Top-k results to retrieve (default: 20)
#   --llm       Use LLM-based reranker instead of cross-encoder
#   --dspy      Use DSPy modules (reformulator + picker)
#   --no-cache  Disable resource caching
#   --quick     Quick test mode (10 examples)
#   --full      Full evaluation (100 examples)
#   --baseline  Only run baseline reranking evaluation
#   --system    Only run full system comparison
#   --graphs    Only generate graphs from existing results
#   -h, --help  Show this help message
#
# Examples:
#   ./run_evaluation.sh --quick                # Quick test (10 examples)
#   ./run_evaluation.sh -n 100 --llm          # 100 examples with LLM reranker
#   ./run_evaluation.sh --full --dspy         # Full evaluation with DSPy
#   ./run_evaluation.sh --baseline            # Only baseline reranking
################################################################################

set -e  # Exit on error

# Default configuration
NUM_EXAMPLES=50
K=20
USE_LLM_RERANKER=""
USE_DSPY=""
NO_CACHE=""
RUN_BASELINE=true
RUN_SYSTEM=true
RUN_GRAPHS=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}$1${NC}"
    echo "================================================================================"
    echo ""
}

# Help message
show_help() {
    cat << EOF
Comprehensive Baseline Evaluation Runner

Usage: $0 [options]

Options:
  -n NUM      Number of examples to evaluate (default: 50)
  -k K        Top-k results to retrieve (default: 20)
  --llm       Use LLM-based reranker instead of cross-encoder
  --dspy      Use DSPy modules (reformulator + picker)
  --no-cache  Disable resource caching
  --quick     Quick test mode (10 examples)
  --full      Full evaluation (100 examples)
  --baseline  Only run baseline reranking evaluation
  --system    Only run full system comparison
  --graphs    Only generate graphs from existing results
  -h, --help  Show this help message

Examples:
  $0 --quick                # Quick test (10 examples)
  $0 -n 100 --llm          # 100 examples with LLM reranker
  $0 --full --dspy         # Full evaluation with DSPy
  $0 --baseline            # Only baseline reranking

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            NUM_EXAMPLES="$2"
            shift 2
            ;;
        -k)
            K="$2"
            shift 2
            ;;
        --llm)
            USE_LLM_RERANKER="--llm-reranker"
            shift
            ;;
        --dspy)
            USE_DSPY="--use-dspy"
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --quick)
            NUM_EXAMPLES=10
            shift
            ;;
        --full)
            NUM_EXAMPLES=100
            shift
            ;;
        --baseline)
            RUN_BASELINE=true
            RUN_SYSTEM=false
            RUN_GRAPHS=false
            shift
            ;;
        --system)
            RUN_BASELINE=false
            RUN_SYSTEM=true
            RUN_GRAPHS=false
            shift
            ;;
        --graphs)
            RUN_BASELINE=false
            RUN_SYSTEM=false
            RUN_GRAPHS=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Display configuration
log_section "EVALUATION CONFIGURATION"
log_info "Number of examples: $NUM_EXAMPLES"
log_info "Top-k results: $K"
log_info "LLM reranker: $([ -n "$USE_LLM_RERANKER" ] && echo "Enabled" || echo "Disabled (using cross-encoder)")"
log_info "DSPy modules: $([ -n "$USE_DSPY" ] && echo "Enabled" || echo "Disabled")"
log_info "Cache: $([ -n "$NO_CACHE" ] && echo "Disabled" || echo "Enabled")"
log_info ""
log_info "Running:"
log_info "  - Baseline reranking: $([ "$RUN_BASELINE" = true ] && echo "Yes" || echo "No")"
log_info "  - System comparison: $([ "$RUN_SYSTEM" = true ] && echo "Yes" || echo "No")"
log_info "  - Graph generation: $([ "$RUN_GRAPHS" = true ] && echo "Yes" || echo "No")"

# Check if dataset is configured
if [ -z "$DATASET_DIR" ]; then
    log_warning "DATASET_DIR environment variable not set"
    log_info "Checking .env file..."
fi

# Start timestamp
START_TIME=$(date +%s)

# ============================================================================
# 1. BASELINE RERANKING EVALUATION
# ============================================================================

if [ "$RUN_BASELINE" = true ]; then
    log_section "Step 1: Baseline Reranking Evaluation"
    log_info "Running: evaluate_baselines_with_reranking.py"
    log_info "This evaluates BM25, E5, and SPECTER with and without LLM reranking"

    BASELINE_CMD="uv run python evaluate_baselines_with_reranking.py --num-examples $NUM_EXAMPLES --k $K $USE_LLM_RERANKER $NO_CACHE"

    log_info "Command: $BASELINE_CMD"
    echo ""

    if eval $BASELINE_CMD; then
        log_success "Baseline reranking evaluation completed!"
        log_info "Results saved to: baseline_reranking_results/"
    else
        log_error "Baseline reranking evaluation failed!"
        exit 1
    fi
fi

# ============================================================================
# 2. FULL SYSTEM COMPARISON
# ============================================================================

if [ "$RUN_SYSTEM" = true ]; then
    log_section "Step 2: Full System Comparison"
    log_info "Running: compare_baselines_vs_system.py"
    log_info "This compares individual baselines vs the full system"

    SYSTEM_CMD="uv run python compare_baselines_vs_system.py --num-examples $NUM_EXAMPLES --k $K $USE_LLM_RERANKER $USE_DSPY $NO_CACHE"

    log_info "Command: $SYSTEM_CMD"
    echo ""

    if eval $SYSTEM_CMD; then
        log_success "System comparison completed!"
        log_info "Results saved to: comparison_results/"
    else
        log_error "System comparison failed!"
        exit 1
    fi
fi

# ============================================================================
# 3. GENERATE INDIVIDUAL GRAPHS
# ============================================================================

if [ "$RUN_GRAPHS" = true ]; then
    log_section "Step 3: Generate Individual Graphs"
    log_info "Running: generate_individual_graphs.py"
    log_info "This creates individual visualization graphs from results"

    # Check if baseline results exist
    if [ ! -f "baseline_reranking_results/full_results.json" ]; then
        log_warning "baseline_reranking_results/full_results.json not found"
        log_warning "Skipping graph generation. Run with --baseline first."
    else
        if uv run python generate_individual_graphs.py; then
            log_success "Individual graphs generated!"
            log_info "Graphs saved to: baseline_reranking_results/individual_graphs/"
        else
            log_error "Graph generation failed!"
            exit 1
        fi
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

log_section "EVALUATION COMPLETE"
log_success "All evaluations completed successfully!"
log_info "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
log_info "Results locations:"

if [ "$RUN_BASELINE" = true ]; then
    echo "  📁 baseline_reranking_results/"
    echo "     - full_results.json"
    echo "     - summary_table.csv"
    echo "     - visualization.png"
fi

if [ "$RUN_SYSTEM" = true ]; then
    echo "  📁 comparison_results/"
    echo "     - full_results.json"
    echo "     - aggregated_results.json"
    echo "     - comparison_table.csv"
    echo "     - bar_chart_comparison.png"
    echo "     - radar_chart_comparison.png"
    echo "     - heatmap_comparison.png"
    echo "     - improvement_chart.png"
fi

if [ "$RUN_GRAPHS" = true ] && [ -d "baseline_reranking_results/individual_graphs" ]; then
    echo "  📁 baseline_reranking_results/individual_graphs/"
    echo "     - Method comparisons (3 graphs)"
    echo "     - Recall curves (3 graphs)"
    echo "     - Improvement charts (3 graphs)"
    echo "     - Metric views (4 graphs)"
    echo "     - Overall MRR (1 graph)"
fi

echo ""
log_info "To view results:"
log_info "  - Open PNG files in baseline_reranking_results/ and comparison_results/"
log_info "  - Review CSV files for detailed metrics"
log_info "  - Check JSON files for raw results"

echo ""
echo "================================================================================"
