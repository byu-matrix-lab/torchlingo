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

# Run the training script with full 20 epochs (no early stopping)
python examples/train.py \
    --data-dir "PATH_TO_DATA_DIR" \
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
