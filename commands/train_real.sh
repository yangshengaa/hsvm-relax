#!/bin/bash
#SBATCH --job-name=hsvm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-06:00
#SBATCH -p seas_compute
#SBATCH --mem=240GB
#SBATCH --array=0-5%6
#SBATCH -o out/real_%A_%a.out
#SBATCH -e out/real_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shengyang@fas.harvard.edu

# activate env
source ~/.bashrc
conda activate base

tag="remote"

# select penalty
if [ $((SLURM_ARRAY_TASK_ID % 3)) == 0 ]; then
    C=0.1
elif [ $((SLURM_ARRAY_TASK_ID % 3)) == 1 ]; then
    C=1.0
else
    C=10.0
fi

# select datasets
if [ $((SLURM_ARRAY_TASK_ID / 3)) == 0 ]; then
    datasets=("football" "karate" "krumsiek" "moignard" "cifar")
else
    datasets=("myeloidprogenitors" "olsson" "paul" "polbooks" "fashion-mnist")
fi

seed=0
for data in "${datasets[@]}"; do
    # run euclidean 
    python3 src/train.py --data $data --seed $seed --model euclidean --C $C --tag $tag --verbose --dump

    # run projected gradient descent
    python3 src/train.py --data $data --seed $seed --model gd --C $C --tag $tag --verbose --dump

    # run SDP 
    python3 src/train.py --data $data --seed $seed --model sdp --C $C --dump --tag $tag --verbose --dump

    # run moment relaxation
    python3 src/train.py --data $data --seed $seed --model moment --C $C --dump --tag $tag --verbose --dump
done 