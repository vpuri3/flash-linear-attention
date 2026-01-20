#!/bin/bash

# #============================#
# # Advanced Queue (4 GPUs)
# #============================#
# #SBATCH --account=lkara
# #SBATCH --partition=advanced
# #SBATCH --qos=adv_4gpu_qos
# #SBATCH --job-name=FLARE
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:4        # 4 GPUs per node
# #SBATCH --ntasks-per-node=4 # 4 tasks per node → 1 per GPU
# #SBATCH --cpus-per-task=26  # give each rank 26 CPU cores (208 / 8 = 26) 

# #============================#
# # Advanced Queue (3 GPU)
# #============================#
# #SBATCH --account=lkara
# #SBATCH --partition=advanced
# #SBATCH --qos=adv_4gpu_qos
# #SBATCH --job-name=FLARE
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:3
# #SBATCH --ntasks-per-node=3
# #SBATCH --cpus-per-task=26

# #============================#
# # Advanced Queue (2 GPU)
# #============================#
# #SBATCH --account=lkara
# #SBATCH --partition=advanced
# #SBATCH --qos=adv_4gpu_qos
# #SBATCH --job-name=FLARE
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:2
# #SBATCH --ntasks-per-node=2
# #SBATCH --cpus-per-task=26

# #============================#
# # Advanced Queue (1 GPU)
# #============================#
# #SBATCH --account=lkara
# #SBATCH --partition=advanced
# #SBATCH --qos=adv_4gpu_qos
# #SBATCH --job-name=FLARE
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=26

# #============================#
# # General Queue
# #============================#
# #SBATCH --account=lkara
# #SBATCH --partition=general
# #SBATCH --qos=general_qos
# #SBATCH --job-name=FLARE
# #SBATCH --time=12:00:00
# #SBATCH --gres=gpu:1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=26

# #============================#
# # General Queue (1 GPU)
# #============================#
# #SBATCH --account=lkara
# #SBATCH --partition=preempt
# #SBATCH --qos=preempt_qos
# #SBATCH --job-name=FLARE
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=26

#============================#
# Preempt Queue (4 GPU)
#============================#
#SBATCH --account=lkara
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --job-name=FLARE
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26

#============================#
# Setup
#============================#
echo "Setting up environment"
source ~/.bash_profile
cd /project/community/$(whoami)/FLARE-dev.py
source .venv/bin/activate

#============================#
# Run interactively with
#   $ srun --pty --overlap --jobid <job_id> bash
sleep 24h
#============================#

# your execution code here

#============================#
wait
#============================#
#