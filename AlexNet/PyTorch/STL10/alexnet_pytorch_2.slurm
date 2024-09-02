#!/bin/bash
#SBATCH --job-name=pytorch_cnn_STL10
#SBATCH --output=pytorch_cnn_STL10_%j.out
#SBATCH --error=pytorch_cnn_STL10_%j.err
#SBATCH --partition=gpu2v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

# Print start time
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

# Load Anaconda module
echo "Loading Anaconda module..."
module load anaconda3

# Activate your environment
echo "Activating conda environment..."
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate myvm

# Verify the environment
echo "Conda environment:"
conda env list

# Verify GPU availability
echo "GPU information:"
nvidia-smi

# Run your Python script
echo "Running PyTorch CNN script..."
python pytorch_improved_cnn_STL10.py

# Print end time
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"