**RebalancedCV**
=============
<!-- badges: start -->
[![main](https://github.com/korem-lab/RebalancedCV/actions/workflows/main.yml/badge.svg)](https://github.com/korem-lab/RebalancedCV/actions/workflows/main.yml)
<!-- badges: end -->
<img src='vignettes/RLOOCV-logo.png' align="right" height="250" />

This is a python package designed to facilitate correcting for distributional bias during cross valiation.  It was recently shown that removing a fraction of a dataset into a testing fold can artificially create a shift in label averages across training folds that is inversely correlated with that of their corresponding test folds. We have demonstrated that most machine learning models' results suffer from this bias, which this package resolves by subsampling points from within the trianing set to remove any differences in label average across training folds. to begin using RebalancedCV, we recommend reading it's [documentation pages](https://korem-lab.github.io/RebalancedCV/).


All classes from this package provide train/test indices to split data in train/test sets while rebalancing the training set to account for distributional bias. This package is designed to enable automated rebalancing for the cross-valition implementations in formats similar to scikit-learn's `LeaveOneOut`, `StratifiedKFold`, and `LeavePOut`, through the `RebalancedCV` classes `RebalancedLeaveOneOut`, `RebalancedLeaveOneOutRegression`, `RebalancedKFold`, and `RebalancedLeavePOut`. These Rebalanced classes are designed to work in the exact same code structure and implementation use cases as their scikit-learn equivalents, with the only difference being a subsampling within the provided training indices.


**Installation**
-------------------
```bash
pip install git+https://github.com/korem-lab/RebalancedCV.git
```
The dependencies for RebalancedCV are python, numpy, and scikit-learn. Is has been developped and tested using python 3.6 - 3.12. Only standard hardware is required for RebalancedCV. The typical install time for RebalancedCV is less that 15 seconds. 


**Example**
-----------------
We demonstrate the following snippet of code to utilize out rebalanced leave-one-out implementation, using an observation matrix `X` and a binary outcome vector `y`. We demonstrate it using scikit-learn's `LogisticRegressionCV`, although this can be replaced with any training/tuning/predicting scheme. The expected runtime for the RebalancedCV package's operations are less than 5 seconds, wihle the overall example is expected to complete in less than one minute on most machines.

```python
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut
from rebalancedcv import RebalancedLeaveOneOut
from sklearn.metrics import roc_auc_score
np.random.seed(1)

## given some random `X` matrix, and a `y` binary vector
X = np.random.rand(100, 10)
y = np.random.rand(100) > 0.5

## Leave-one-out evaluation
loo = LeaveOneOut()
loocv_predictions = [ LogisticRegressionCV()\
                                .fit(X[train_index], y[train_index])\
                                .predict_proba(X[test_index]
                                            )[:, 1][0]
              for train_index, test_index in loo.split(X, y) ]

## Since all the data is random, a fair evaluation
## should yield au auROC close to 0.5
print('Leave One Out auROC: {:.2f}'\
              .format( roc_auc_score(y, loocv_predictions) ) )

## Rebalanced leave-one-out evaluation
rloo = RebalancedLeaveOneOut()
rloocv_predictions = [ LogisticRegressionCV()\
                                .fit(X[train_index], y[train_index])\
                                .predict_proba(X[test_index]
                                            )[:, 1][0]
              for train_index, test_index in rloo.split(X, y) ]

## Since all the data is random, a fair evaluation
## should yield au auROC close to 0.5
print('Rebalanceed Leave-one-out auROC: {:.2f}'\
              .format(  roc_auc_score(y, rloocv_predictions) ) )
```

    Leave One Out auROC: 0.00
    Rebalanceed Leave-one-out auROC: 0.48


As demontrated in this example, neglecting to account for distributional bias in the cross-valiation classes can greatly decrease evaluated model performance. For more details on why this happens, please refer to Austin, G.I. et al. “Distributional bias compromises leave-one-out cross-validation” (2024). https://arxiv.org/abs/2406.01652. 


We note that the example's code structure appraoch would apply to this package's other `RebalancedKFold` and `RebalancedLeavePOut` classes.

**Classes**
---------

### RebalancedLeaveOneOut

Provides train/test indices to split data in train/test sets with rebalancing to ensure that all training folds have identical class balances. Each sample is used once as a test set, while the remaining samples form the training set. See sklearn.model_selection.LeaveOneOut for more details on Leave-one-out cross-validation. 

##### **Parameters**
No parameters are used for this class 

### RebalancedKFold

Provides train/test indices to split data in `n_splits` folds, with rebalancing to ensure that all training folds have identical class balances. Each sample is only ever used within a single test fold. `RebalancedKFold` uses the following parameters, which are the same as the scikit-learn `StratifiedKFold` parameters (see sklearn.model_selection.StratifiedKFold for more details):

##### **Parameters**
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.


### RebalancedLeavePOut

Provides train/test indices to split data in train/test sets with rebalancing to ensure that all training folds have identical class balances. This cross-validation tests on all distinct samples of size p, while a remaining n - 2p samples form the training set in each iteration, with an additional `p` samples used to subsamples from within the training set.
(see sklearn.model_selection.LeavePOut for more details).

##### **Parameters**
     p : int
        Size of the test sets. Must be strictly less than one half of the number of samples.
        

### RebalancedLeaveOneOutRegression

Designed for regression tasks. Provides train/test indices to split data in train/test sets with rebalancing to ensure that all training folds have similar labels balances. Each sample is used once as a test set, while the remaining samples form the training set. See sklearn.model_selection.LeaveOneOut for more details on Leave-one-out cross-validation. 

##### **Parameters**
No parameters are used for this class 

**Parameters for `.split()` method**
----------
All three of this package's classes use the `split` method, which all use the following parameters.
`X` : array-like of shape (n_samples, n_features); Training data, where `n_samples` is the number of samples and `n_features` is the number of features.

`y` : array-like of shape (n_samples,); The target variable for supervised learning problems.  At least two observations per class are needed for RebalancedLeaveOneOut

`groups` : array-like of shape (n_samples,), default=None; Group labels for the samples used while splitting the dataset into
    train/test set.
    
`seed` : Integer, default=None; can be specified to enforce consistency in the subsampling

**Yields**
-------
`train_index` : ndarray
    The training set indices for that split.
    
`test_index` : ndarray
    The testing set indices for that split.



**Citation**
-------
Austin, G.I. et al. “Distributional bias compromises leave-one-out cross-validation” (2024). https://arxiv.org/abs/2406.01652
