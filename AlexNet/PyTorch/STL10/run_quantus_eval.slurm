#!/bin/bash
#SBATCH --job-name=quantus_xai_eval
#SBATCH --output=quantus_xai_eval_%j.out
#SBATCH --error=quantus_xai_eval_%j.err
#SBATCH --partition=gpu2v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

# Set and print the working directory
WORK_DIR="$PWD/quantus_evaluation"
mkdir -p $WORK_DIR
echo "Working directory: $WORK_DIR"

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

# Create a timestamped results directory
RESULTS_DIR="$WORK_DIR/quantus_results_$(date '+%Y%m%d_%H%M%S')"
mkdir -p $RESULTS_DIR
echo "Results will be saved in: $RESULTS_DIR"

# Export the results directory for the Python script to use
export RESULTS_DIR

# Run your Python script
echo "Running Quantus XAI evaluation script..."
python quantus_captum_evaluation.py

# Print end time
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results saved in: $RESULTS_DIR"