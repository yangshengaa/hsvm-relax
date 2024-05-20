# Hyperbolic SVM Relaxation

SDP and Moment Relaxation of Hyperbolic SVM

## Environment

Install the following packages

```bash
python -m venv hsvm_relax
source ./hsvm_relax/bin/activate

# install packages
pip install matplotlib torch toml mosek cvxpy SumOfSquares
```

for `mosek`, one can get an academic license for free from [https://www.mosek.com/products/academic-licenses/](https://www.mosek.com/products/academic-licenses/)

To run experiments, simply follow the bash scripts in [commands/](commands/). One may directly run the bash scripts or adapt accordingly.
