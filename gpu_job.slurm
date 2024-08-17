#!/bin/bash
#SBATCH --job-name=mqa887_gpu_job
#SBATCH --output=mqa887_gpu_job_%j.out
#SBATCH --error=mqa887_gpu_job_%j.err
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
source activate myvm

# Verify the environment
echo "Conda environment:"
conda env list

# Verify GPU availability
echo "GPU information:"
nvidia-smi

# Run your Python script
echo "Running Python script..."
python pre_main.py

# Print end time
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"