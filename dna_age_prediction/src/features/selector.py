"""
Feature selection module for identifying biologically relevant CpG markers.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings


class FeatureSelector:
    """
    Select most relevant features for age prediction.
    
    Methods:
    - Variance-based selection
    - Correlation-based selection
    - Mutual information
    - Univariate regression
    """
    
    def __init__(self, method: str = "f_regression", n_features: int = 1000):
        """
        Initialize feature selector.
        
        Parameters
        ----------
        method : str
            'variance', 'f_regression', 'mutual_info'
        n_features : int
            Number of features to select
        """
        self.method = method
        self.n_features = n_features
        self.selector = None
        self.selected_indices = None
        self.scores = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """
        Fit feature selector.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
            
        Returns
        -------
        self : FeatureSelector
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        if X.shape[1] == 0:
            raise ValueError("Feature matrix cannot be empty")
        
        n_features = min(self.n_features, X.shape[1])
        
        if self.method == "f_regression":
            self.selector = SelectKBest(f_regression, k=n_features)
        elif self.method == "mutual_info":
            self.selector = SelectKBest(mutual_info_regression, k=n_features)
        elif self.method == "variance":
            # Variance-based selection
            variances = np.var(X, axis=0)
            self.selected_indices = np.argsort(variances)[-n_features:]
            self.is_fitted = True
            return self
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.selector.fit(X, y)
        self.selected_indices = self.selector.get_support(indices=True)
        self.scores = self.selector.scores_
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Select features from data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        X_selected : np.ndarray
            Data with selected features only
        """
        if not self.is_fitted:
            raise ValueError("Selector must be fitted before transform")
        
        return X[:, self.selected_indices]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns
        -------
        scores : np.ndarray
            Importance scores for selected features
        """
        if self.scores is None:
            raise ValueError("Scores not available for this selection method")
        
        return self.scores[self.selected_indices]
    
    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        return self.selected_indices
    
    @staticmethod
    def remove_correlated_features(X: np.ndarray, threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove highly correlated features.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        threshold : float
            Correlation threshold for removal
            
        Returns
        -------
        X_filtered : np.ndarray
            Data with correlated features removed
        keep_indices : np.ndarray
            Indices of retained features
        """
        corr_matrix = np.corrcoef(X.T)
        keep_indices = []
        removed = set()
        
        for i in range(corr_matrix.shape[0]):
            if i in removed:
                continue
            
            keep_indices.append(i)
            
            # Find highly correlated features
            for j in range(i + 1, corr_matrix.shape[0]):
                if j not in removed and abs(corr_matrix[i, j]) > threshold:
                    removed.add(j)
        
        return X[:, keep_indices], np.array(keep_indices)
