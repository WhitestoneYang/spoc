# SPOC

This package is aimed at molecular descriptor generation, data processing, model training, and hyper-parameter optimization.

1. Summarizes various molecular descriptor generation methods provided by different tools/packages, including RDKit, CDK, Openbabel, Pubchem, Deepchem, etc. It's easy for batch generation.
2. Data pre-processing and splitting.
3. Modeling training and hyperparameter optimization by leveraging Scikit-Learn, XGBoost, and LightGBM, more machine learning, and neural network methods will be included/wrapped in the future.

## Dependencies

SPOC currently supports Python >= 3.6 and requires these packages on any condition.

## - [Anaconda](https://www.anaconda.com/)[Java SE Development Kit](https://www.oracle.com)

- [OpenBabel](http://openbabel.org)
- [deepchem](https://github.com/deepchem/deepchem)
- [tensorflow](https://www.tensorflow.org/install)
- [scikit-learn](https://scikit-learn.org)
- [pubchempy](https://github.com/mcs07/PubChemPy)
- [lightgbm](https://github.com/microsoft/LightGBM)
- [xgboost](https://github.com/dmlc/xgboost)
- [bayesian-optimization](https://github.com/fmfn/BayesianOptimization)

## Installation


### Method 1: conda 

```bash
# Clone project
git clone git@github.com:WhitestoneYang/spoc.git # or other released or tagged version.

# conda installation
bash - i conda_installation.sh
```

### Method 2: docker

```bash
# docker build
docker build --progress=plain -t spoc .

# docker run
docker run -v $(pwd):/workspace/ --network host -it spoc
```

## Usage

1. Please refer the [tests](./tests) for descriptor generation examples, including single and multiple molecular descriptor generation examples
2. Please refer the [examples](./examples) for 1) molecular descriptor generation; 2) data processing; 3) model training; 4) hyper parameter optimization workflow.

## Citing SPOC

If you have used SPOC in your research, please [**cite our paper**](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/cphc.202200255).
