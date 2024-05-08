"""
train Soft-Margin Hyperbolic SVM models
"""

# load packages
import os 
from time import time
import json
from typing import Tuple
import argparse
import numpy as np
from sklearn.model_selection import KFold

# load files
from utils import load_config, name_exp

parser = argparse.ArgumentParser()

# data related
parser.add_argument('--data', default="gaussian", type=str, help='the data to train/test on. Use "gaussian" for simulated data')
parser.add_argument('--N', default=200, type=int, help='number of points to simulate per class')
parser.add_argument('--dim', default=2, type=int, help='the dimension of simulated dataset')
parser.add_argument('--scale', default=0.5, type=float, help='the standard deviation of gaussian data')
parser.add_argument("--seed", default=730, type=int, help='the random seed')
parser.add_argument('--K', default=2, type=int, help='the number of classes to generate')
parser.add_argument("--folds", default=5, type=int, help='the number of folds')

# model 
parser.add_argument('--model', default="sdp", type=str, help='select model')
parser.add_argument('--C', default=1, type=float, help='the penalty strength, must be positive')
parser.add_argument('--lr', default=0.001, type=float, help='the learning rate for PGD')
parser.add_argument('--kappa', default=2, type=int, help='the relaxation order')
parser.add_argument("--refine", default=False, action='store_true', help='turn on to enable local refinement')
parser.add_argument("--refine-method", default="COBYLA", type=str, help='the method for local refinement')
parser.add_argument('--verbose', default=False, action='store_true', help='print mosek logging')
parser.add_argument("--multi-class", default='ovr', type=str, help='the multiclass training scheme', choices=['ovr', 'ovo'])

# technical
parser.add_argument('--tag', default='exp', type=str, help='the tag for loading data')
parser.add_argument('--dump', default=False, action='store_true', help='use this flag to dump trained parameters')

args = parser.parse_args()

# get path 
paths = load_config(args.tag)

# name the experiment
exp_name = name_exp(args)
print(f"==== {exp_name} =====")
model_path = os.path.join(paths['model_dir'], exp_name)
result_path = os.path.join(paths['result_dir'], exp_name)
os.makedirs(model_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

def load_data() -> Tuple[np.ndarray]:
    """get data"""
    if args.data == 'gaussian':
        from data import hyperbolic_gaussian
        X, y = hyperbolic_gaussian(args.dim, args.K, args.N, args.seed, args.scale)
    else:
        from data import load_lorentz_features
        X, y = load_lorentz_features(args.data, paths["data_dir"])
    return X, y

def load_model():
    """get model"""
    if args.model.lower() == 'gd':
        from models import HyperbolicSVMSoft
        model = HyperbolicSVMSoft(args.C, args.lr, multi_class=args.multi_class,seed=args.seed)
    elif args.model.lower() == 'sdp':
        from models import HyperbolicSVMSoftSDP
        model = HyperbolicSVMSoftSDP(args.C, multi_class=args.multi_class, refine=args.refine, refine_method=args.refine_method)
    elif args.model.lower() == "moment":
        from models import HyperbolicSVMSoftSOSSparsePrimal
        model = HyperbolicSVMSoftSOSSparsePrimal(args.C, multi_class=args.multi_class, refine=args.refine, refine_method=args.refine_method)
    elif args.model.lower () == 'euclidean':
        # treats hyperbolic features as living in the ambient space
        from models import EuclideanSVMSoft
        model = EuclideanSVMSoft(args.C, multi_class=args.multi_class, fit_intercept=False)
    else:
        raise NotImplementedError(f"model {args.model} not implemented")
    return model

def train(X, y):
    """train loop, log K-fold validation acc and loss"""
    # set up KFold validation
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    info = {}
    start_time = time()
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        fold_info = {}
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]

        # initialize a new model
        model = load_model()

        # kappa argument only used for SOS 
        model.fit(train_X, train_y, verbose=args.verbose, kappa=args.kappa)

        # get train test acc 
        train_acc = (model.predict(train_X) == train_y).mean()
        test_acc = (model.predict(test_X) == test_y).mean()

        fold_info["train_acc"] = train_acc 
        fold_info['test_acc'] = test_acc

        # get loss 
        if args.model.lower() != 'euclidean':
            train_loss = model.get_train_loss()
            test_loss = model.get_test_loss(X, y, args.C)

            fold_info['train_loss'] = train_loss
            fold_info['test_loss'] = test_loss

        # get optimality gap 
        if args.model.lower() == 'sdp' or args.model.lower() == 'moment':
            fold_info['optimality_gap'] = model.get_optimality_gap()

        info[f"fold_{i + 1}"] = fold_info

        # dump model parameters
        if args.dump:
            model.save(model_path, f"fold_{i + 1}")

    end_time = time()
    time_elapse = end_time - start_time
    info["time"] = time_elapse

    return info

def main():
    X, y = load_data()
    info = train(X, y)

    # dump training information to file
    with open(os.path.join(result_path, 'info.json'), 'w') as f:
        json.dump(info, f)


if __name__ == '__main__':
    main()
