#!/bin/bash
#SBATCH --job-name=hsvm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-06:00
#SBATCH -p seas_compute
#SBATCH --mem=192GB
#SBATCH --array=0-35%12
#SBATCH -o out/sim_%A_%a.out
#SBATCH -e out/sim_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shengyang@fas.harvard.edu

# activate env
source ~/.bashrc
conda activate base

tag="remote"

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
    dim=2
elif [ $((SLURM_ARRAY_TASK_ID / 12)) == 1 ]; then
    dim=3
else 
    dim=4
fi


for seed in $(seq 1 40); do 
    # run euclidean 
    python3 src/train.py --data gaussian --N 200 --dim $dim --scale $scale --K 2 --seed $seed --model euclidean --C $C --tag $tag --verbose

    # run projected gradient descent
    python3 src/train.py --data gaussian --N 200 --dim $dim --scale $scale --K 2 --seed $seed --model gd --C $C --tag $tag --verbose

    # run SDP 
    python3 src/train.py --data gaussian --N 200 --dim $dim --scale $scale --K 2 --seed $seed --model sdp --C $C --dump --tag $tag --verbose

    # run moment relaxation
    python3 src/train.py --data gaussian --N 200 --dim $dim --scale $scale --K 2 --seed $seed --model moment --C $C --dump --tag $tag --verbose
done 
