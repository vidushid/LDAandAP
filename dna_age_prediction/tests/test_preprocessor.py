import numpy as np
from src.data.preprocessor import DataPreprocessor

def test_preprocessor_runs():
    X = np.random.rand(50, 100)
    pre = DataPreprocessor()
    X_t = pre.fit_transform(X)
    assert X_t.shape[0] == 50
