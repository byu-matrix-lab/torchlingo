#!/bin/bash
#SBATCH --job-name=torchlingo_train_full
#SBATCH --output=slurm/logs/train_torchlingo_%j.out
#SBATCH --error=slurm/logs/train_torchlingo_%j.out
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=matrix

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate torchlingo_py310

# Verify we're in the right directory (project root)
cd /home/myl15/torchlingo/torchlingo

# Run the training script with full 20 epochs (no early stopping)
python examples/train.py \
    --data-dir "/home/myl15/groups/grp_mtlab/nobackup/archive/all-data/91-cleaned/all-church-data/Chinese T/filtered/" \
    --vocab-size 8000 \
    --num-epochs 500 \
    --batch-size 128 \
    --learning-rate 1e-4 \
    --patience 12 \
    --scheduler plateau \
    --src-lang eng \
    --tgt-lang cmn \
    --experiment-name torchlingo_eng_cmn_plateau_convergence \
    --device cuda \
    --data-format txt \
    --val-interval 1000 \
    --save-interval 2000 \
