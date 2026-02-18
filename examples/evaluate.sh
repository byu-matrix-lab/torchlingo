#!/bin/bash
#SBATCH --job-name=torchlingo_eval
#SBATCH --output=slurm/logs/eval_torchlingo_%j.out
#SBATCH --error=slurm/logs/eval_torchlingo_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=matrix

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate torchlingo_py310

# Verify we're in the right directory (project root)
cd /home/myl15/torchlingo/torchlingo

# Run the evaluation script
python examples/evaluate.py \
    --checkpoint "checkpoints/{args.experiment_name}/model_best.pt" \
    --test-file "runs/test.tsv" \
    --src-sp-model "runs/sp_eng.model" \
    --tgt-sp-model "runs/sp_cmn.model" \
    --d-model 512 \
    --n-heads 8 \
    --num-encoder-layers 6 \
    --num-decoder-layers 6 \
    --d-ff 2048 \
    --dropout 0.1 \
    --max-seq-length 256 \
    --batch-size 64 \
    --num-samples 10 \
    --src-col src \
    --tgt-col tgt \
    --output-dir "evaluation_results/torchlingo_eng_cmn_baseline_new_package" \
    --device cuda
