"""
Data loader module for GSE40279 and other DNA methylation datasets.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import GEOparse
except ImportError:
    GEOparse = None


class DataLoader:
    """
    Load and manage DNA methylation data from various sources.
    
    Supports:
    - Local CSV/TSV files
    - GEO (Gene Expression Omnibus) datasets
    - Processed methylation matrices
    """
    
    def __init__(self, data_path: str = "data/raw/GSE40279", dataset_id: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        data_path : str
            Path to data directory
        dataset_id : str, optional
            GEO dataset ID (e.g., 'GSE40279')
        """
        self.data_path = data_path
        self.dataset_id = dataset_id
        self.X = None
        self.y = None
        self.metadata = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load methylation data and age labels.
        
        Returns
        -------
        X : np.ndarray
            Methylation data (samples x CpG sites)
        y : np.ndarray
            Age labels (chronological age)
        """
        # Try loading from local files first
        if os.path.exists(self.data_path):
            return self._load_from_local()
        
        # Try GEO download if dataset_id is provided
        if self.dataset_id and GEOparse is not None:
            return self._load_from_geo()
        
        raise FileNotFoundError(
            f"Data not found at {self.data_path} and no valid GEO dataset specified"
        )
    
    def _load_from_local(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from local CSV/TSV files."""
        # Load methylation matrix
        methylation_file = os.path.join(self.data_path, "methylation_matrix.csv")
        phenotype_file = os.path.join(self.data_path, "phenotype_data.csv")
        
        if not os.path.exists(methylation_file):
            # Look for alternative naming
            csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
            if csv_files:
                methylation_file = os.path.join(self.data_path, csv_files[0])
            else:
                raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        # Load methylation data
        X = pd.read_csv(methylation_file, index_col=0)
        
        # Load phenotype/age data
        if os.path.exists(phenotype_file):
            phenotype = pd.read_csv(phenotype_file, index_col=0)
            y = phenotype['age'].values
            self.metadata = phenotype
        else:
            # If no separate phenotype file, look in methylation file columns
            if 'age' in X.columns:
                y = X['age'].values
                X = X.drop('age', axis=1)
            else:
                raise FileNotFoundError(f"Age data not found in {phenotype_file}")
        
        self.X = X.values
        self.y = y
        
        print(f"Loaded data: {self.X.shape[0]} samples, {self.X.shape[1]} CpG sites")
        return self.X, self.y
    
    def _load_from_geo(self) -> Tuple[np.ndarray, np.ndarray]:
        """Download and load data from GEO database."""
        print(f"Downloading {self.dataset_id} from GEO...")
        
        gse = GEOparse.get_GEO(self.dataset_id, destdir="./data/raw/")
        
        # Extract methylation matrix and metadata
        # This is dataset-specific and may need adjustment
        X = gse.pivot_samples('VALUE').values
        y = np.array([float(gse.metadata['characteristics_ch1'][i].split(': ')[1]) 
                      for i in range(len(gse.metadata['geo_accession']))])
        
        self.X = X
        self.y = y
        
        print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def get_feature_names(self) -> Optional[np.ndarray]:
        """Return CpG site names/IDs."""
        if isinstance(self.X, pd.DataFrame):
            return self.X.columns.values
        return None
    
    def get_sample_names(self) -> Optional[np.ndarray]:
        """Return sample IDs."""
        if isinstance(self.X, pd.DataFrame):
            return self.X.index.values
        return None
    
    def get_metadata(self) -> Optional[pd.DataFrame]:
        """Return metadata if available."""
        return self.metadata
    
    @staticmethod
    def create_sample_data(n_samples: int = 100, n_features: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sample synthetic methylation data for testing.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        n_features : int
            Number of CpG sites
            
        Returns
        -------
        X : np.ndarray
            Synthetic methylation matrix
        y : np.ndarray
            Synthetic age labels
        """
        np.random.seed(42)
        
        # Generate synthetic methylation data (values between 0 and 1)
        X = np.random.rand(n_samples, n_features)
        
        # Generate age labels correlated with some features
        y = 40 + 0.5 * np.mean(X[:, :10], axis=1) * 40 + np.random.normal(0, 5, n_samples)
        y = np.clip(y, 20, 80)  # Clip to realistic age range
        
        return X, y
