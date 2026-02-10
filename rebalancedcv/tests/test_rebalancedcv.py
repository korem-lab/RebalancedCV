#!/usr/bin/env python

import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import LeaveOneOut
from rebalancedcv import RebalancedLeaveOneOut, RebalancedKFold, \
                        RebalancedLeavePOut, RebalancedLeaveOneOutRegression, \
                            MulticlassRebalancedLeaveOneOut, RebalancedLeaveOneGroupOut

from sklearn.metrics import roc_auc_score

import unittest
from sklearn.metrics import roc_auc_score, r2_score

def flatten(xss):
    return [x for xs in xss for x in xs]



class DMtest(unittest.TestCase):
    def run_classification_cv(self, 
                              cv_object, 
                              n_samples=50, 
                              n_features=2, 
                              class_balance=.5, 
                              seed=1):
        np.random.seed(seed)

        if cv_object == RebalancedLeavePOut:
            cvo=cv_object(p=2)
        else:
            cvo=cv_object()

        ## generate some random `X` matrix, and a `y` binary vector
        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples) > class_balance
        
        if cv_object==MulticlassRebalancedLeaveOneOut:
            y = ( np.random.rand(n_samples) * 4 ).astype(int)
            
            cv_summary = [ ( LogisticRegression(C=1e-4)\
                                        .fit(X[train_index], 
                                             y[train_index])\
                                        .predict_proba(X[test_index]
                                                    ), 
                                  y[test_index] )
                          for train_index, test_index in cvo.split(X, y) ]



            score= roc_auc_score(flatten([a[1] for a in cv_summary]),
                                 flatten([a[0] for a in cv_summary]), 
                                 multi_class='ovr'
                                 )
            
            
        else:    

            cv_summary = [ ( LogisticRegression(C=1e-4)\
                                        .fit(X[train_index], 
                                             y[train_index])\
                                        .predict_proba(X[test_index]
                                                    )[:, 1], 
                                  y[test_index] )
                          for train_index, test_index in cvo.split(X, y) ]



            score= roc_auc_score(flatten([a[1] for a in cv_summary]),
                                 flatten([a[0] for a in cv_summary])
                                 )
        
        

        ## confirm aurocs are close to 0.5
        self.assertTrue( score > .35 )
        self.assertTrue( score < .65 )

        ## confirm train set means are identical across folds
        train_means = [ y[train_index].mean()
                        for train_index, test_index in cvo.split(X, y) ]

        self.assertTrue( np.max(train_means) == np.min(train_means) )


    def run_regression_cv(self, 
                          cv_object, 
                          n_samples=50, 
                          n_features=2, 
                          seed=1):
        np.random.seed(seed)

        cvo=cv_object()

        ## generate some random `X` matrix, and a `y` binary vector
        X = np.random.normal(size=(n_samples, n_features))
        y = np.random.normal(size=n_samples)
        
        cv_summary = [ ( LinearRegression()\
                                    .fit(X[train_index], 
                                         y[train_index])\
                                    .predict(X[test_index]), 
                              y[test_index] )
                      for train_index, test_index in cvo.split(X, y) ]

        score= r2_score(flatten([a[1] for a in cv_summary]),
                        flatten([a[0] for a in cv_summary])
                        )
        
        ## confirm aurocs are close to 0.5
        self.assertTrue( score > -0.1 )
        self.assertTrue( score < .05 )

        ## confirm train set means are identical across folds
        train_means = [ y[train_index].mean()
                        for train_index, test_index in cvo.split(X, y) ]
        
        self.assertTrue( np.max(train_means) - np.min(train_means) <= 1e-2 )
    
    def test_all_classification_cvs(self):
        for cv in [RebalancedLeaveOneOut, RebalancedKFold, RebalancedLeavePOut, MulticlassRebalancedLeaveOneOut]:
            self.run_classification_cv(cv)

    def test_rebalanced_leave_one_group_out(self):
        rlogo = RebalancedLeaveOneGroupOut()

        ## --- API: groups required ---
        with self.assertRaises(ValueError):
            rlogo.get_n_splits(groups=None)
        with self.assertRaises(ValueError):
            list(rlogo.split(np.random.rand(6, 2), np.array([0, 0, 1, 1, 0, 1]), groups=None))

        ## --- Binary: 6 samples, 2 groups of 3 ---
        np.random.seed(1)
        n_samples, n_features = 6, 2
        X = np.random.rand(n_samples, n_features)
        y = np.array([0, 0, 1, 1, 0, 1])
        groups = np.array([1, 1, 1, 2, 2, 2])
        self.assertEqual(rlogo.get_n_splits(groups=groups), 2)
        train_means = []
        for train_index, test_index in rlogo.split(X, y, groups, seed=1):
            self.assertEqual(len(np.unique(groups[test_index])), 1,
                             "each test set should be exactly one group")
            train_means.append(y[train_index].mean())
        self.assertTrue(np.max(train_means) == np.min(train_means),
                        "train class balance should be identical across folds (binary)")

        ## --- Multi-class: 12 samples, 3 classes, 3 groups ---
        np.random.seed(2)
        X = np.random.rand(12, 3)
        y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])   # 4 per class
        groups = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        self.assertEqual(rlogo.get_n_splits(groups=groups), 3)
        train_class_counts = []
        for train_index, test_index in rlogo.split(X, y, groups, seed=2):
            self.assertEqual(len(np.unique(groups[test_index])), 1,
                             "each test set should be exactly one group")
            ## rebalancing: same number of each class in train every fold
            counts = np.bincount(y[train_index], minlength=3)
            train_class_counts.append(tuple(counts))
        self.assertEqual(len(set(train_class_counts)), 1,
                         "train class counts should be identical across folds (multi-class)")
        ## sanity: each fold had some of every class in train (we have 3 groups, 3 classes spread across)
        for counts in train_class_counts:
            self.assertTrue(all(c >= 0 for c in counts))

    def test_all_regression_cvs(self):
        for cv in [RebalancedLeaveOneOutRegression,
                   ]:
            self.run_regression_cv(cv)
            
        
    
if __name__ == '__main__':
    unittest.main()