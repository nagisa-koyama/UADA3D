#!/bin/bash
#SBATCH --job-name=multigpu          # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks-per-node=4          # total number of tasks per node
#SBATCH --cpus-per-task=8            # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                    # total memory per node
#SBATCH --gres=gpu:4                 # number of allocated gpus per node
#SBATCH --partition=a6000_ada        # GPU node partition
#SBATCH --output=logs/output_%j_a6000_ada_multigpu.txt
#SBATCH --error=logs/error_%j_a6000_ada_multigpu.txt
#SBATCH --time=99:00:00

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

# Multi-GPU training (4 GPUs, DDP via torchrun + --launcher pytorch)
# batch_size must be divisible by n_gpus (4) and by 2 (src/tgt split), so multiples of 8.
# Typical: --batch_size 24 -> 6 per GPU
#
# To add a new experiment, copy one of the lines below, uncomment it, and adjust the arguments.

SINGULARITY="singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif"
TORCHRUN="torchrun --nproc_per_node=4"

# $SINGULARITY $TORCHRUN adaptive_train.py --launcher pytorch --cfg_file cfgs/kitti2nuscenes_models/second-rospm-C.yaml --batch_size 24 --wandb_notes "" --extra_tag 20260503_multigpu
# $SINGULARITY $TORCHRUN adaptive_train.py --launcher pytorch --cfg_file cfgs/kitti2nuscenes_models/centerpoint-rospm-C.yaml --batch_size 24 --wandb_notes "" --extra_tag 20260503_multigpu
# $SINGULARITY $TORCHRUN adaptive_train.py --launcher pytorch --cfg_file cfgs/nuscenes2kitti_models/centerpoint-rospm-C.yaml --batch_size 24 --wandb_notes "" --extra_tag 20260503_multigpu
$SINGULARITY $TORCHRUN adaptive_train.py --launcher pytorch --cfg_file cfgs/nuscenes2kitti_models/second-rospm-C.yaml --batch_size 24 --wandb_notes "da second nuscenes2kitti 5 epochs conditional-only uniform range multi-gpu" --extra_tag 20260503_3_multigpu
