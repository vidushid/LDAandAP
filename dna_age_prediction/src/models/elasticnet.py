"""
ElasticNet regression model for age prediction.
"""

import numpy as np
from sklearn.linear_model import ElasticNet as SklearnElasticNet
from typing import Tuple, Optional
import warnings


class ElasticNetModel:
    """
    ElasticNet regression model for DNA methylation-based age prediction.
    
    Combines L1 (Lasso) and L2 (Ridge) regularization.
    Good for feature interpretation and handling multicollinearity.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 0.0001,
        random_state: int = 42
    ):
        """
        Initialize ElasticNet model.
        
        Parameters
        ----------
        alpha : float
            Regularization strength
        l1_ratio : float
            Mix of L1 and L2 (0 = Ridge, 1 = Lasso)
        fit_intercept : bool
            Whether to fit intercept
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        random_state : int
            Random seed
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.model = SklearnElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
        
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNetModel':
        """
        Fit ElasticNet model.
        
        Parameters
        ----------
        X : np.ndarray
            Features (samples x features)
        y : np.ndarray
            Target (age labels)
            
        Returns
        -------
        self : ElasticNetModel
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
        
        if X.shape[1] != self.model.coef_.shape[0]:
            raise ValueError(
                f"Expected {self.model.coef_.shape[0]} features, got {X.shape[1]}"
            )
        
        return self.model.predict(X)
    
    def get_coefficients(self) -> np.ndarray:
        """Get model coefficients (feature weights)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.coef_
    
    def get_intercept(self) -> float:
        """Get model intercept."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.intercept_
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance (absolute coefficient values).
        
        Returns
        -------
        importance : np.ndarray
            Absolute coefficient values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return np.abs(self.model.coef_)
    
    def get_nonzero_features(self) -> np.ndarray:
        """Get indices of non-zero coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return np.where(self.model.coef_ != 0)[0]
    
    def get_model_params(self) -> dict:
        """Get model parameters."""
        return {
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
