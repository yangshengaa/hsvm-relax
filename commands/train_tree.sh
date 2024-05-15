#!/bin/bash
#SBATCH --job-name=hsvm_tree
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-06:00
#SBATCH -p seas_compute
#SBATCH --mem=120GB
#SBATCH --array=0-2%3
#SBATCH -o out/tree/sim_%A_%a.out
#SBATCH -e out/tree/sim_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shengyang@fas.harvard.edu

# activate env
source ~/.bashrc
conda activate base

tag="tree"
seed=0

# select penalty
if [ $((SLURM_ARRAY_TASK_ID)) == 0 ]; then
    C=0.1
elif [ $((SLURM_ARRAY_TASK_ID)) == 1 ]; then
    C=1.0
else
    C=10.0
fi

# script for training tree 
for data in "tree_1" "tree_2" "tree_3"; do 
    # run euclidean 
    python3 src/train.py --data $data --seed $seed --model euclidean --C $C --tag $tag --verbose

    # run projected gradient descent
    python3 src/train.py --data $data --seed $seed --model gd --C $C --tag $tag --verbose

    # run SDP 
    python3 src/train.py --data $data --seed $seed --model sdp --C $C --dump --tag $tag --verbose

    # run moment relaxation
    python3 src/train.py --data $data --seed $seed --model moment --C $C --dump --tag $tag --verbose
done
