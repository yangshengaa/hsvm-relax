#!/bin/bash
#SBATCH --job-name=hsvm_gaussian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-06:00
#SBATCH -p seas_compute
#SBATCH --mem=180GB
#SBATCH --array=0-35%9
#SBATCH -o out/gaussian/sim_%A_%a.out
#SBATCH -e out/gaussian/sim_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shengyang@fas.harvard.edu

# activate env
source ~/.bashrc
conda activate base

tag="gaussian"

# select penalty
if [ $(((SLURM_ARRAY_TASK_ID % 12) / 4)) == 0 ]; then
    C=0.1
elif [ $(((SLURM_ARRAY_TASK_ID % 12) / 4)) == 1 ]; then
    C=1.0
else
    C=10.0
fi

# select scale 
if [ $((SLURM_ARRAY_TASK_ID % 4)) == 0 ]; then
    scale=0.4
elif [ $((SLURM_ARRAY_TASK_ID % 4)) == 1 ]; then
    scale=0.6
elif [ $((SLURM_ARRAY_TASK_ID % 4)) == 2 ]; then
    scale=0.8
else 
    scale=1.0
fi

# select dimension
if [ $((SLURM_ARRAY_TASK_ID / 12)) == 0 ]; then
    K=2
elif [ $((SLURM_ARRAY_TASK_ID / 12)) == 1 ]; then
    K=3
else 
    K=5
fi


for seed in $(seq 1 40); do 
    # run euclidean 
    python3 src/train.py --data gaussian --N 100 --dim 2 --scale $scale --K $K --seed $seed --model euclidean --C $C --tag $tag --verbose

    # run projected gradient descent
    python3 src/train.py --data gaussian --N 100 --dim 2 --scale $scale --K $K --seed $seed --model gd --C $C --tag $tag --verbose

    # run SDP 
    python3 src/train.py --data gaussian --N 100 --dim 2 --scale $scale --K $K --seed $seed --model sdp --C $C --dump --tag $tag --verbose

    # run moment relaxation
    python3 src/train.py --data gaussian --N 100 --dim 2 --scale $scale --K $K --seed $seed --model moment --C $C --dump --tag $tag --verbose
done 
