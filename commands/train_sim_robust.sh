#!/bin/bash
#SBATCH --job-name=hsvm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-06:00
#SBATCH -p seas_compute
#SBATCH --mem=900GB
#SBATCH --array=0-26%9
#SBATCH -o out/real_%A_%a.out
#SBATCH -e out/real_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shengyang@fas.harvard.edu

# scripts for training robust SDP
# activate env
source ~/.bashrc
conda activate base

tag="robust"

# select penalty
if [ $(((SLURM_ARRAY_TASK_ID % 9) / 3)) == 0 ]; then
    C=0.1
elif [ $(((SLURM_ARRAY_TASK_ID % 9) / 3)) == 1 ]; then
    C=1.0
else
    C=10.0
fi

# select scale 
if [ $((SLURM_ARRAY_TASK_ID % 3)) == 0 ]; then
    scale=0.4
elif [ $((SLURM_ARRAY_TASK_ID % 3)) == 1 ]; then
    scale=0.7
else 
    scale=1.0
fi

# select dimension
if [ $((SLURM_ARRAY_TASK_ID / 9)) == 0 ]; then
    K=2
elif [ $((SLURM_ARRAY_TASK_ID / 9)) == 1 ]; then
    K=3
else 
    K=4
fi


for seed in $(seq 1 10); do 

    # run projected gradient descent
    python3 src/train.py --data gaussian --N 200 --dim 2 --scale $scale --K $K --seed $seed --model gd --C $C --tag $tag --verbose --multi-class ovo

    # run SDP 
    python3 src/train.py --data gaussian --N 200 --dim 2 --scale $scale --K $K --seed $seed --model sdp --C $C --tag $tag --verbose --multi-class ovo

    for rho in "0.005" "0.01" "0.02"; do 
        # run robust SDP (l-1)
        python3 src/train.py --data gaussian --N 200 --dim 2 --scale $scale --K $K --seed $seed --model sdp_1 --C $C --tag $tag --verbose --multi-class ovo --rho $rho

        # run robust SDP (l-inf)
        python3 src/train.py --data gaussian --N 200 --dim 2 --scale $scale --K $K --seed $seed --model sdp_inf --C $C --tag $tag --verbose --multi-class ovo --rho $rho
    done
done 
