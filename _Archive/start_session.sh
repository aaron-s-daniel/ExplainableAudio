#!/bin/bash

# Create a temporary script for the Slurm job
cat << 'EOF' > temp_job.sh
#!/bin/bash
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

# Load Anaconda module
module load anaconda3

# Activate conda environment
source activate myvm

# Keep the job running
while true; do
    sleep 60
done
EOF

# Submit the job and capture the job ID
job_id=$(sbatch --parsable temp_job.sh)

# Clean up the temporary script
rm temp_job.sh

echo "Job submitted with ID: $job_id"
echo "Waiting for job to start..."

# Wait for the job to start
while true; do
    status=$(squeue -h -j "$job_id" -o %t)
    if [ "$status" = "R" ]; then
        break
    fi
    sleep 5
done

echo "Job is running. Connecting to the job..."

# Connect to the running job
srun --jobid "$job_id" --pty /bin/bash -i