RebalancedCV
-------------------------
<img src='vignettes/RLOOCV-logo.png' align="right" height="250" />

This is a packaged designed to facilitate correcting for distributional bias in leave-one-out. The design of this class is to mimic the same structure and scikit-learn's `LeaveOneOut`.

Installation
-------------------
```bash
pip install git+https://github.com/korem-lab/RebalancedCV.git
```

Example
-----------------
We demonstrate the following snippet of code to utilize out rebalanced leave-one-out implementation, using an observation matrix `X` and a binary outcome vector `y`. We demonstrate it using scikit-learn's `LogisticRegressionCV`, although this can be replaced with any training/tuning/predicting scheme. 

```python
from sklearn.linear_model import LogisticRegressionCV
from rebalancedcv import RebalancedLeaveOneOut

## given some `X` matrix, and a `y` binary vector
rloo = RebalancedLeaveOneOut()
cv_predictions = [ LogisticRegressionCV()\
                                .fit(X[train_index], y[train_index])\
                                .predict_proba(X[test_index]
                                            )[:, 1][0]
              for train_index, test_index in rloo.split(X, y) ]
roc_auc_score(y, cv_predictions)
```

Per our analyses (see **Austin et al. in prep** for more details), our approach resolves the common issue of distributional bias that can be observed in standard `LeaveOneOut` implementations. 


Parameters
----------
`X` : array-like of shape (n_samples, n_features); Training data, where `n_samples` is the number of samples and `n_features` is the number of features.

`y` : array-like of shape (n_samples,); The target variable for supervised learning problems.  At least two observations per class are needed for RebalancedLeaveOneOut

`groups` : array-like of shape (n_samples,), default=None; Group labels for the samples used while splitting the dataset into
    train/test set.
    
`seed` : Integer, default=None; can be specified to enforce consistency in the subsampling

Yields
-------
`train_index` : ndarray
    The training set indices for that split.
`test_index` : ndarray
    The testing set indices for that split.
