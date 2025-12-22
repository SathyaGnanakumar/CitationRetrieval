#!/bin/bash
# Quick-start script for DSPy incremental optimization
#
# This script runs DSPy optimization in batches with default settings.
# Modify parameters as needed for your use case.
#
# Requirements:
# - OPENAI_API_KEY environment variable must be set
# - Training data must be prepared (run data_prep.py first)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "DSPy Incremental Optimization"
echo "=================================="

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY environment variable not set${NC}"
    echo ""
    echo "DSPy optimization requires OpenAI API access."
    echo "Set your API key:"
    echo "  export OPENAI_API_KEY=your_key_here"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ OpenAI API key found${NC}"

# Default parameters (can be overridden by command-line arguments)
BATCH_SIZE=${1:-100}
MAX_BATCHES=${2:-10}
TEACHER_MODEL=${3:-gpt-5-mini-2025-08-07}
STUDENT_MODEL=${4:-gpt-5-mini-2025-08-07}
MODULE=${5:-rerank}
OPTIMIZER=${6:-bootstrap}

echo ""
echo "Configuration:"
echo "  Batch size: $BATCH_SIZE examples"
echo "  Max batches: $MAX_BATCHES"
echo "  Teacher model: $TEACHER_MODEL"
echo "  Student model: $STUDENT_MODEL"
echo "  Module: $MODULE"
echo "  Optimizer: $OPTIMIZER"
echo ""

# Check if data exists
DATA_DIR="data"
if [ ! -f "$DATA_DIR/train.json" ] || [ ! -f "$DATA_DIR/val.json" ]; then
    echo -e "${YELLOW}⚠️  Training data not found in $DATA_DIR/${NC}"
    echo ""
    echo "Please prepare data first:"
    echo "  python data_prep.py \\"
    echo "    --dataset ../../../corpus/scholarcopilot/scholar_copilot_eval_data_1k.json \\"
    echo "    --output-dir data/"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Training data found${NC}"
echo ""

# Check for existing checkpoint
CHECKPOINT="optimized/checkpoint.json"
if [ -f "$CHECKPOINT" ]; then
    echo -e "${YELLOW}⚠️  Found existing checkpoint at $CHECKPOINT${NC}"
    echo "The optimizer will resume from the last completed batch."
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Run optimization
echo ""
echo "=================================="
echo "Starting optimization..."
echo "=================================="
echo ""

python incremental_optimizer.py \
    --train-path "$DATA_DIR/train.json" \
    --val-path "$DATA_DIR/val.json" \
    --test-path "$DATA_DIR/test.json" \
    --batch-size "$BATCH_SIZE" \
    --max-batches "$MAX_BATCHES" \
    --teacher-model "$TEACHER_MODEL" \
    --student-model "$STUDENT_MODEL" \
    --module "$MODULE" \
    --optimizer "$OPTIMIZER"

echo ""
echo "=================================="
echo -e "${GREEN}✅ Optimization complete!${NC}"
echo "=================================="
echo ""
echo "Results saved to: optimized/"
echo ""
echo "To view results:"
echo "  cat optimized/checkpoint.json"
echo ""
echo "To use the optimized model, load it in your code:"
echo "  module.load('optimized/batch_N_module.pkl')"
echo ""
