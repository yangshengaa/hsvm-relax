# Hyperbolic SVM Relaxation

SDP and Moment Relaxation of Hyperbolic SVM.

For details, see our preprint: [https://arxiv.org/abs/2405.17198](https://arxiv.org/abs/2405.17198).

## Environment

Install the following packages

```bash
# create a virtual env (recommend python >= 3.8)
python -m venv hsvm_relax
source ./hsvm_relax/bin/activate

# install packages
pip install matplotlib scikit-learn toml mosek cvxpy SumOfSquares
```

`mosek` is a commercial solver. One can get an academic license for free from [https://www.mosek.com/products/academic-licenses/](https://www.mosek.com/products/academic-licenses/).

## Running Instructions

to reproduce results, we first specify the relative paths with respect to the root of the project in [config.toml](comfig.toml):

- `data_dir`: path to read data;
- `model_dir`: path to store model parameters;
- `result_dir`: path to dump Kfold train test results.

along with the associated tags. For experiment tag, see the following snippet also in [config.toml](config.toml)

```toml
["exp"]
data_dir='data/'
model_dir='model/'
result_dir='result/'
```

To run experiments, simply follow the bash scripts in [commands/](commands/). One may directly run the bash scripts (if with access to a Slurm system) or adapt accordingly.

One example is as follows: use moment relaxation with $C = 10.0$ on krumsiek dataset with a 5 fold train test split (default) can be called by

```bash
# run the following at root
python src/train.py --data krumsiek --model moment --C 10. --verbose --dump
```

where `--verbose` indicates printing the interior point progress summary and `--dump` indicates saving the trained model parameters. See [src/train.py](src/train.py) for a full lists of parameters.

## Acknowledgement

If you find this code useful, please consider citing our preprint:

    @article{yang2024convex,
      title={Convex Relaxation for Solving Large-Margin Classifiers in Hyperbolic Space},
      author={Yang, Sheng and Liu, Peihan and Pehlevan, Cengiz},
      journal={arXiv preprint arXiv:2405.17198},
      year={2024}
    }
