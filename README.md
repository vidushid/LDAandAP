# DNA-Based Age Prediction

A machine learning project for predicting chronological age from DNA methylation (CpG) profiles using ElasticNet and Random Forest models.

## Overview

This project implements and compares multiple machine learning approaches to predict human age from DNA methylation data. By analyzing CpG site methylation patterns, we develop regression models that can estimate biological age, with potential applications in gerontology research and epigenetic aging studies.

### Key Features

- **Data Processing**: Comprehensive preprocessing of DNA methylation data (GSE40279 - Hannum dataset)
- **Feature Selection**: Multiple feature selection techniques to identify biologically relevant CpG markers
- **Model Comparison**: ElasticNet vs Random Forest regression models
- **Evaluation Metrics**: R², MAE, RMSE, and cross-validation analysis
- **Biological Interpretation**: Feature importance analysis to identify age-related methylation markers

## Dataset

- **Source**: GSE40279 (Hannum et al., 2013)
- **Type**: DNA methylation microarray data (Illumina 450k)
- **Samples**: ~656 healthy individuals
- **Features**: CpG methylation sites
- **Target**: Chronological age

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dna-age-prediction.git
   cd dna-age-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package**
   ```bash
   pip install -e .
   ```

## Usage

### Quick Start

```python
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.elasticnet import ElasticNetModel
from src.evaluation.metrics import evaluate_model

# Load data
loader = DataLoader(data_path='data/raw/GSE40279/')
X, y = loader.load_data()

# Preprocess
preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(X)

# Train model
model = ElasticNetModel(alpha=1.0, l1_ratio=0.5)
model.fit(X_processed, y)

# Evaluate
metrics = evaluate_model(model, X_processed, y)
print(metrics)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_models.py -v
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pyyaml >= 5.4.0


## References

1. Hannum, G., et al. (2013). Genome-wide methylation profiles reveal quantitative views on mammalian aging. *Molecular Cell*
2. Horvath, S. (2013). DNA methylation age of human tissues and cell types. *Genome Biology*
3. Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12:2825-2830, 2011
