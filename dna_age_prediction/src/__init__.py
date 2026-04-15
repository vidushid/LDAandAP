"""
DNA-based Age Prediction Package

A machine learning package for predicting chronological age from DNA methylation profiles.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.selector import FeatureSelector
from src.models.elasticnet import ElasticNetModel
from src.models.random_forest import RandomForestModel
from src.evaluation.metrics import evaluate_model

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "FeatureSelector",
    "ElasticNetModel",
    "RandomForestModel",
    "evaluate_model",
]
