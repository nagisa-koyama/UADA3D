# Multi-GPU Training Setup Report for UADA3D

**Date:** 2026-05-03  
**Experiment:** `nuscenes2kitti / second-rospm-C`  
**Jobs:** SLURM `#15505` (failed), `#15506` (running)

---

## 1. Background

`adaptive_train.py` already contained full PyTorch DDP support:
- `--launcher` argument (`none` / `pytorch` / `slurm`)
- `init_dist_pytorch` / `init_dist_slurm` initialization
- Per-GPU batch size splitting
- `nn.parallel.DistributedDataParallel` wrapping
- Distributed samplers for both source and target dataloaders
- W&B / TensorBoard logging gated to rank 0

However, the existing `run_experiments.sh` only used a single GPU (`--gres=gpu:1`) and plain `python3`, leaving DDP unused.

---

## 2. New Script: `run_experiments_multigpu.sh`

Created [`run_experiments_multigpu.sh`](run_experiments_multigpu.sh) as a multi-GPU counterpart to `run_experiments.sh`.

### SBATCH Configuration

| Parameter | Value |
|---|---|
| `--nodes` | 1 |
| `--ntasks-per-node` | 4 |
| `--cpus-per-task` | 8 |
| `--mem` | 64G |
| `--gres` | `gpu:4` |
| `--partition` | `a6000_ada` |

### Batch Size Constraints

The script uses `--batch_size 24` (total), which satisfies both constraints in `adaptive_train.py`:
- `batch_size % total_gpus == 0` → `24 % 4 == 0` ✓
- `batch_size_per_gpu % 2 == 0` → `6 % 2 == 0` ✓ (50/50 source/target split per GPU)

### Script Structure

Uses `SINGULARITY` and `TORCHRUN` variables to avoid repetition. New experiments can be added by copying and uncommenting a line:

```bash
SINGULARITY="singularity exec --nv --bind /home/koyama/data/:/storage ..."
TORCHRUN="torchrun --nproc_per_node=4"

# $SINGULARITY $TORCHRUN adaptive_train.py --launcher pytorch --cfg_file ... --batch_size 24 ...
$SINGULARITY $TORCHRUN adaptive_train.py --launcher pytorch --cfg_file cfgs/nuscenes2kitti_models/second-rospm-C.yaml --batch_size 24 ...
```

---

## 3. Job #15505 — Failed

### Error

```
Duplicate GPU detected : rank 3 and rank 0 both on CUDA device 2a000
torch.distributed.DistBackendError: NCCL error — invalid usage
```

All 4 ranks were binding to the same GPU (cuda:0).

### Root Cause

`torchrun` (PyTorch ≥ 1.10) injects `LOCAL_RANK` as an **environment variable**, not as a `--local_rank` CLI argument (that was the legacy `torch.distributed.launch` behavior). Since `args.local_rank` defaulted to `0` for all processes, `init_dist_pytorch` called `torch.cuda.set_device(0)` on every rank, causing NCCL to see 4 processes on the same GPU.

### Fix — `adaptive_train.py`

Added one line in `main()` before calling `init_dist_pytorch`:

```python
# torchrun sets LOCAL_RANK as an env var (not a CLI arg) in PyTorch >= 1.10
args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
```

This ensures each process reads its correct local rank (0, 1, 2, 3) from the environment.

---

## 4. Job #15506 — Running

After the fix, job `#15506` was submitted and reached:

```
INFO  **********************Start training nuscenes2kitti_models/second-rospm-C(20260503_3_multigpu)**********************
```

Training is running successfully on 4 GPUs.

W&B run: https://wandb.ai/nagisa/uada3d/runs/wrmwm5bp

---

## 5. Usage

```bash
# Submit multi-GPU job
sbatch run_experiments_multigpu.sh

# Add a new experiment: uncomment/copy a line in run_experiments_multigpu.sh
# $SINGULARITY $TORCHRUN adaptive_train.py --launcher pytorch \
#   --cfg_file cfgs/<experiment>.yaml \
#   --batch_size 24 --wandb_notes "..." --extra_tag ...
```
