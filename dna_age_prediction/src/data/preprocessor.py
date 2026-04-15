"""
Data preprocessing module for DNA methylation data.

Includes handling missing values, normalization, and quality control.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import warnings


class DataPreprocessor:
    """
    Preprocess DNA methylation data.
    
    Operations:
    - Remove missing values
    - Filter low-variance features
    - Normalize/standardize data
    - Handle outliers
    """
    
    def __init__(
        self,
        missing_threshold: float = 0.1,
        variance_threshold: float = 0.01,
        normalization_method: str = "quantile",
        remove_outliers: bool = False
    ):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        missing_threshold : float
            Fraction of missing values allowed per feature (0-1)
        variance_threshold : float
            Minimum variance threshold for feature retention
        normalization_method : str
            'quantile', 'zscore', 'minmax', or None
        remove_outliers : bool
            Whether to remove outlier samples
        """
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.normalization_method = normalization_method
        self.remove_outliers = remove_outliers
        
        self.scaler = None
        self.feature_mask = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'DataPreprocessor':
        """
        Fit preprocessor on data.
        
        Parameters
        ----------
        X : np.ndarray
            Methylation data (samples x features)
            
        Returns
        -------
        self : DataPreprocessor
        """
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Input data cannot be empty")
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Filter low-variance features
        self.feature_mask = self._get_variance_mask(X)
        
        # Initialize scaler
        if self.normalization_method == "quantile":
            self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        elif self.normalization_method == "zscore":
            self.scaler = StandardScaler()
        elif self.normalization_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        self.is_fitted = True
        return self
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit on data and return transformed data.
        
        Parameters
        ----------
        X : np.ndarray
            Methylation data
            
        Returns
        -------
        X_transformed : np.ndarray
            Preprocessed methylation data
        """
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Parameters
        ----------
        X : np.ndarray
            Methylation data
            
        Returns
        -------
        X_transformed : np.ndarray
            Preprocessed methylation data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Input data cannot be empty")
        
        X = X.copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Apply feature mask
        if self.feature_mask is not None:
            X = X[:, self.feature_mask]
        
        # Normalize
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        # Remove outliers
        if self.remove_outliers:
            X = self._remove_outliers(X)
        
        return X
    
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values in data."""
        X = X.copy()
        
        # Replace inf with nan
        X[np.isinf(X)] = np.nan
        
        # Count missing values per feature
        missing_fraction = np.sum(np.isnan(X), axis=0) / X.shape[0]
        
        # Remove features with too many missing values
        valid_features = missing_fraction <= self.missing_threshold
        X = X[:, valid_features]
        
        # For remaining missing values, use mean imputation
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_means[i]
        
        return X
    
    def _get_variance_mask(self, X: np.ndarray) -> np.ndarray:
        """Get boolean mask for features with sufficient variance."""
        variance = np.var(X, axis=0)
        return variance > self.variance_threshold
    
    def _remove_outliers(self, X: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """Remove outlier samples using z-score."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        
        z_scores = np.abs((X - mean) / (std + 1e-8))
        outlier_mask = np.all(z_scores < n_std, axis=1)
        
        return X[outlier_mask]
    
    def get_feature_stats(self) -> dict:
        """Return preprocessing statistics."""
        return {
            'missing_threshold': self.missing_threshold,
            'variance_threshold': self.variance_threshold,
            'normalization_method': self.normalization_method,
            'n_features_retained': np.sum(self.feature_mask) if self.feature_mask is not None else None,
            'is_fitted': self.is_fitted
        }
