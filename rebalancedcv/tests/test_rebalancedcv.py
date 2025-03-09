#!/usr/bin/env python

import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import LeaveOneOut
from rebalancedcv import RebalancedLeaveOneOut, RebalancedKFold, \
                        RebalancedLeavePOut, RebalancedLeaveOneOutRegression

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
        for cv in [RebalancedLeaveOneOut, RebalancedKFold, RebalancedLeavePOut]:
            self.run_classification_cv(cv)
            
    def test_all_regression_cvs(self):
        for cv in [RebalancedLeaveOneOutRegression,
                   ]:
            self.run_regression_cv(cv)
            
        
    
if __name__ == '__main__':
    unittest.main()