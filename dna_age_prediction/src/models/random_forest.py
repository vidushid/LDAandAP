"""
Random Forest regression model for age prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Optional
import warnings


class RandomForestModel:
    """
    Random Forest regression model for DNA methylation-based age prediction.
    
    Advantages:
    - Captures non-linear relationships
    - Provides feature importance measures
    - Robust to outliers
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest model.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int
            Minimum samples required to split
        min_samples_leaf : int
            Minimum samples required at leaf
        max_features : str
            Features to consider at each split
        random_state : int
            Random seed
        n_jobs : int
            Number of parallel jobs (-1 = all CPUs)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """
        Fit Random Forest model.
        
        Parameters
        ----------
        X : np.ndarray
            Features (samples x features)
        y : np.ndarray
            Target (age labels)
            
        Returns
        -------
        self : RandomForestModel
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples to fit model")
        
        if X.shape[1] == 0:
            raise ValueError("Feature matrix cannot be empty")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict age from methylation data.
        
        Parameters
        ----------
        X : np.ndarray
            Features
            
        Returns
        -------
        predictions : np.ndarray
            Predicted ages
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if X.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"Expected {self.model.n_features_in_} features, got {X.shape[1]}"
            )
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns
        -------
        importance : np.ndarray
            Importance for each feature
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.feature_importances_
    
    def get_top_features(self, n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top N most important features.
        
        Parameters
        ----------
        n : int
            Number of top features
            
        Returns
        -------
        indices : np.ndarray
            Feature indices
        importances : np.ndarray
            Importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importances = self.get_feature_importance()
        top_indices = np.argsort(importances)[-n:][::-1]
        
        return top_indices, importances[top_indices]
    
    def get_model_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'is_fitted': self.is_fitted
        }
