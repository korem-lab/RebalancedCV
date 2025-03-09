from sklearn.utils import indexable, check_random_state
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples, check_array, column_or_1d
import numpy as np

import numbers
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.multiclass import type_of_target
from abc import ABCMeta, abstractmethod
from itertools import chain, combinations

def flatten(xss):
    return [x for xs in xss for x in xs]

class RebalancedLeaveOneOutRegression(BaseCrossValidator):
    """Rebalanced Leave-One-Out cross-validator

    Provides train/test indices to split data in train/test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples are used to form the training set, 
    with subsampling to ensure consistent class balances across all splits.

    This class is designed to have the same functionality and 
    implementation structure as scikit-learn's ``LeaveOneOut()``
    
    At least two observations per class are needed for `RebalancedLeaveOneOut`

    Examples
    --------
    >>> import numpy as np
    >>> from rebalancedcv import RebalancedLeaveOneOut
    >>> X = np.array([[1, 2, 1, 2], [3, 4, 3, 4]]).T
    >>> y = np.array([1, 2, 1, 2])
    >>> rloo = RebalancedLeaveOneOut()
    >>> for i, (train_index, test_index) in enumerate(rloo.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
        Fold 0:
          Train: index=[2 3]
          Test:  index=[0]
        Fold 1:
          Train: index=[0 3]
          Test:  index=[1]
        Fold 2:
          Train: index=[0 1]
          Test:  index=[2]
        Fold 3:
          Train: index=[0 1]
          Test:  index=[3]
    """
    
    def split(self, X, y, groups=None, seed=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
            
        seed : to enforce consistency in the subsampling

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        
        if seed is not None:
            np.random.seed(seed)
            
        X, y, groups = indexable(X, y, None)

        indices = np.arange(_num_samples(X))
        mean_y = y.mean()
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            tst_pt=y[test_index]
            is_greater = 2*( tst_pt >= mean_y ) - 1
            
            candidate_mask = ( is_greater*y[train_index] >= is_greater * ( mean_y - ( tst_pt - mean_y) ) )* \
                             ( is_greater*y[train_index] <= is_greater*mean_y )
            
            if candidate_mask.sum()>0:
                to_drop=np.where(candidate_mask)[0][ np.argmin(is_greater*\
                                                                   y[train_index][candidate_mask]) ]
                train_index=np.delete(train_index, to_drop)
                
            test_index = indices[test_index]
            yield train_index, test_index


    def _iter_test_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        if n_samples <= 1:
            raise ValueError(
                "Cannot perform LeaveOneOut with n_samples={}.".format(n_samples)
            )
        return range(n_samples)

    def get_n_splits(self, X, y, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : object
            Needed to maintin class balance consistency.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return _num_samples(X)