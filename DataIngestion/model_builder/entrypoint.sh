#!/bin/bash
echo "Checking GPU availability..."
nvidia-smi || echo "No GPU detected"
echo ""
echo "CUDA devices visible to PyTorch:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"
echo ""
exec "$@"