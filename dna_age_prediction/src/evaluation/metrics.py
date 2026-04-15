"""
Model evaluation metrics for age prediction.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings


class ModelMetrics:
    """Calculate and manage model evaluation metrics."""
    
    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² score (coefficient of determination)."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pearson correlation coefficient."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        
        if len(y_true) < 2:
            raise ValueError("Need at least 2 samples for correlation")
        
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        return correlation
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        
        if np.any(y_true == 0):
            warnings.warn("y_true contains zeros, MAPE may be undefined")
        
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Parameters
    ----------
    model : object
        Fitted model with predict method
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    X_train : np.ndarray, optional
        Train features for train metrics
    y_train : np.ndarray, optional
        Train labels for train metrics
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("X_test and y_test must have same number of samples")
    
    if X_test.shape[0] == 0:
        raise ValueError("Test set cannot be empty")
    
    # Test predictions
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'test': {
            'r2': ModelMetrics.r_squared(y_test, y_pred_test),
            'mae': ModelMetrics.mae(y_test, y_pred_test),
            'rmse': ModelMetrics.rmse(y_test, y_pred_test),
            'mse': ModelMetrics.mse(y_test, y_pred_test),
            'correlation': ModelMetrics.correlation(y_test, y_pred_test),
        }
    }
    
    # Train metrics if provided
    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train)
        metrics['train'] = {
            'r2': ModelMetrics.r_squared(y_train, y_pred_train),
            'mae': ModelMetrics.mae(y_train, y_pred_train),
            'rmse': ModelMetrics.rmse(y_train, y_pred_train),
            'mse': ModelMetrics.mse(y_train, y_pred_train),
            'correlation': ModelMetrics.correlation(y_train, y_pred_train),
        }
    
    return metrics


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'r2'
) -> Dict:
    """
    Perform cross-validation.
    
    Parameters
    ----------
    model : object
        Model with sklearn-compatible interface
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric ('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error')
        
    Returns
    -------
    cv_results : dict
        Cross-validation results
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    
    if cv < 2:
        raise ValueError("cv must be >= 2")
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'cv': cv,
        'scoring': scoring
    }

