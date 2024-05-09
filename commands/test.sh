python3 src/train.py --data gaussian --N 200 --dim 2 --scale 0.5 --K 2 --seed 400 --model sdp --C 0.1 --dump
python3 src/train.py --data gaussian --N 200 --dim 2 --scale 0.5 --K 2 --seed 400 --model gd --lr 0.001 --C 0.1 --dump
python3 src/train.py --data gaussian --N 200 --dim 2 --scale 0.5 --K 2 --seed 400 --model moment --C 0.1 --dump
python3 src/train.py --data gaussian --N 200 --dim 2 --scale 0.5 --K 2 --seed 400 --model euclidean --C 0.1 --dump
