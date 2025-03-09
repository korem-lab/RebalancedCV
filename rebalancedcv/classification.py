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

class RebalancedLeaveOneOut(BaseCrossValidator):
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
            
        X, y, groups = indexable(X, y, groups)
        
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            
            ## drop one sample with a `y` different from the test index
            susbample_inds = train_index != train_index[ 
                    np.random.choice( np.where( y[train_index] 
                                                         != y[test_index][0])[0] ) ]
            train_index=train_index[susbample_inds]
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
    
    
class _RebalanacedBaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for RebalancedKFold"""

    @abstractmethod
    @_deprecate_positional_args
    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                'Setting a random_state has no effect since shuffle is '
                'False. You should leave '
                'random_state to its default (None), or set shuffle=True.',
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                .format(self.n_splits, n_samples))

        for train, test in super().split(X, y, groups):
            train_inds = np.sort(
                flatten(
                    np.random.choice(train[self.y_encoded[ train ]==a], 
                                     size=self.train_counts[i], 
                                     replace=False
                                     )
                      for i, a in enumerate( np.unique( self.y_encoded ) )
                     ))
            yield train_inds, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class RebalancedKFold(_RebalanacedBaseKFold):
    """Stratified K-Folds cross-validator with Rebalancing to 
    subsample and ensure consistent class balances across all training splits.
    
    Provides train/test indices to split data in train/test sets.
    
    This class is designed to have the same functionality and 
    implementation structure as scikit-learn's ``StratifiedKFold()`

    This cross-validation object is a variation of StratifiedKFold that returns
    perfectly balanced stratified folds. The folds are made by identically preserving 
    the percentage of samples for each class.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RebalancedKFold
    >>> X = np.array([[1, 2, 1, 2, 1], [3, 4, 3, 4, 3]]).T
    >>> y = np.array([1, 2, 1, 2, 1])
    >>> rkf = RebalancedKFold(n_splits=2)
    >>> rkf.get_n_splits(X, y)
    2
    >>> print(rkf)
    >>> for i, (train_index, test_index) in enumerate(rloo.split(X, y)):
    >>>     print(f"Fold {i}:")
    >>>     print(f"  Train: index={train_index}")
    >>>     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[3 4]
      Test:  index=[0 1 2]
    Fold 1:
      Train: index=[0 1]
      Test:  index=[3 4]

    Notes
    -----
    The implementation is designed to:

    * Generate test sets such that all contain the exact same distribution of
      classes.
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Preserve order dependencies in the dataset ordering, when
      ``shuffle=False``: all samples from class k in some test set were
      contiguous in y, or separated in y by samples from classes other than k.
    * Generate test sets where the smallest and largest differ by at most one
      sample.

    .. versionchanged:: 0.22
        The previous implementation did not follow the last constraint.

    See Also
    --------
    RebalancedLeaveOneOut : Implementation of RLOOCV.
    """
    @_deprecate_positional_args
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]
        self.y_encoded=y_encoded

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (self.n_splits))
        if self.n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is less than n_splits=%d."
                           % (min_groups, self.n_splits)), UserWarning)

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [np.bincount(y_order[i::self.n_splits], minlength=n_classes)
             for i in range(self.n_splits)])
        
        ## the number of training points for each class that can be
        ## used within each training fold
        self.train_counts=y_counts-allocation.max(axis=0)
        

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(y), dtype='i')
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        y = check_array(y, 
                        ensure_2d=False, 
                        dtype=None
                        )
        return super().split(X, y, groups)

    
class RebalancedLeavePOut(BaseCrossValidator):
    """Rebalanced Leave-P-Out cross-validator.

    Provides train/test indices to split data in train/test sets with subsampling within the training set to ensure that all training folds have identical class balances. This cross-validation tests on all distinct samples of size p, while a remaining n - 2p samples form the training set in each iteration, with an additional p samples used to subsamples from within the training set.

    This class is designed to have the same functionality and implementation structure as scikit-learn’s LeavePOut()

    Note: Similarly to what was previously mentioned in scikit-learn’s documentation, RebalancedLeavePOut(p) is NOT equivalent to RebalancedKFold(n_splits=n_samples // p) which creates non-overlapping test sets. Due to the high number of iterations which grows combinatorically with the number of samples this cross-validation method can be very costly.

    At least 1+p observations per class are needed for RebalancedLeavePOut.

    Parameters
    ----------
    p : int
        Size of the test sets. Must be strictly less than the one half of the number of
        samples.

    Examples
    --------
    >>> import numpy as np
    >>> from RebalanvedCV import RebalancedLeavePOut
    >>> X = np.array([[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]]).T
    >>> y = np.array([0,1,0,1,0,1])
    >>> rlpo = RebalancedLeavePOut(2)
    >>> for i, (train_index, test_index) in enumerate(rlpo.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[4 5]
      Test:  index=[0 1]
    Fold 1:
      Train: index=[1 4]
      Test:  index=[0 2]
    Fold 2:
      Train: index=[4 5]
      Test:  index=[0 3]
    Fold 3:
      Train: index=[2 3]
      Test:  index=[0 4]
    ...
    """

    def __init__(self, p):
        self.p = p
        
        
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
            
        X, y, groups = indexable(X, y, groups)

        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]

            n_to_drop = self.p - np.bincount(y[test_index], 
                                             minlength=self.n_classes)
            y_vals_to_drop = np.unique( y.astype(int)) 

            to_drop_inds = flatten( [ 
                        list( np.random.choice(
                                train_index[ y[train_index] == y_vals_to_drop[i] ], 
                                                         size=n_to_drop[i], 
                                                         replace=False) ) 
                                 for i in range(len(n_to_drop)) ] )

            train_index = np.array([ a for a in train_index 
                                     if a not in to_drop_inds] )
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        
        self.n_classes = np.unique(y).shape[0]
        
        if n_samples <= self.p:
            raise ValueError(
                "p={} must be strictly less than the number of samples={}".format(
                    self.p, n_samples
                )
            )
        for combination in combinations(range(n_samples), self.p):
            yield np.array(combination)

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return int(comb(_num_samples(X), self.p, exact=True))


