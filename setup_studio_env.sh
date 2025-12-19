#!/bin/bash
set -e

echo "====================== UMIACS LangGraph Studio Setup ======================"

# -----------------------------------------------------------------------------
# 1. Define environment location
# -----------------------------------------------------------------------------
ENV_DIR="/fs/classhomes/sgnanaku/conda_envs/studio"

echo "[1/10] Creating conda environment at: $ENV_DIR"
mkdir -p "$(dirname $ENV_DIR)"

# Create the env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p $ENV_DIR python=3.11
else
    echo "Environment already exists — continuing."
fi

# -----------------------------------------------------------------------------
# 2. Activate environment
# -----------------------------------------------------------------------------
echo "[2/10] Activating environment"
source ~/.bashrc
conda activate $ENV_DIR

# -----------------------------------------------------------------------------
# 3. Upgrade pip
# -----------------------------------------------------------------------------
echo "[3/10] Upgrading pip"
pip install --upgrade pip

# -----------------------------------------------------------------------------
# 4. Install matching PyTorch Nightly (CUDA 12.4)
#    Required because SPECTER blocks torch < 2.6
# -----------------------------------------------------------------------------
echo "[4/10] Installing PyTorch nightly CUDA 12.4"

pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

# Verify installation
echo "[Torch version check]"
python - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# -----------------------------------------------------------------------------
# 5. Install LangGraph + CLI + API
# -----------------------------------------------------------------------------
echo "[5/10] Installing LangGraph + CLI + inmem backend"
pip install "langgraph[all]" langgraph-api langgraph-cli

# -----------------------------------------------------------------------------
# 6. NLP stack: Transformers, Sentence Transformers, LangChain
# -----------------------------------------------------------------------------
echo "[6/10] Installing Transformers, LangChain, Sentence Transformers"
pip install transformers langchain sentence-transformers accelerate

# -----------------------------------------------------------------------------
# 7. BM25 + utilities
# -----------------------------------------------------------------------------
echo "[7/10] Installing BM25 libraries"
pip install bm25s rank_bm25 PyStemmer python-dotenv scipy scikit-learn numpy

# -----------------------------------------------------------------------------
# 8. CUDA fragmentation fix
# -----------------------------------------------------------------------------
echo "[8/10] Enabling CUDA fragmentation fix"
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc

# -----------------------------------------------------------------------------
# 9. Add env bin to PATH for langgraph CLI
# -----------------------------------------------------------------------------
echo "[9/10] Ensuring langgraph CLI is on PATH"

if ! grep -q "$ENV_DIR/bin" ~/.bashrc; then
    echo "export PATH=\"$ENV_DIR/bin:\$PATH\"" >> ~/.bashrc
fi

# -----------------------------------------------------------------------------
# 10. Final message
# -----------------------------------------------------------------------------
echo "==================== Setup Complete ===================="
echo "To start using Studio:"
echo ""
echo "    conda activate $ENV_DIR"
echo "    langgraph dev --tunnel"
echo ""
echo "========================================================="
