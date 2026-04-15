from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.selector import FeatureSelector
from src.models.elasticnet import ElasticNetModel
from src.models.random_forest import RandomForestModel
from src.evaluation.metrics import evaluate_model

from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def run_training_pipeline(config_path="config/config.yaml"):
    logger = get_logger()
    config = load_config(config_path)

    logger.info("Loading data...")
    loader = DataLoader(
        data_path=config["data"]["data_path"],
        dataset_id=config["data"]["dataset_name"]
    )
    X, y = loader.load_data()

    logger.info("Preprocessing...")
    preprocessor = DataPreprocessor(
        missing_threshold=config["preprocessing"]["missing_value_threshold"],
        variance_threshold=config["preprocessing"]["min_variance"]
    )
    X = preprocessor.fit_transform(X)

    logger.info("Feature selection...")
    selector = FeatureSelector(
        method=config["feature_selection"]["method"],
        n_features=config["feature_selection"]["n_features"]
    )
    X = selector.fit_transform(X, y)

    logger.info("Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["evaluation"]["test_size"],
        random_state=config["evaluation"]["random_state"]
    )

    # ElasticNet
    logger.info("Training ElasticNet...")
    en_model = ElasticNetModel()
    en_model.fit(X_train, y_train)

    en_metrics = evaluate_model(en_model, X_test, y_test)

    # Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train)

    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    results = {
        "elasticnet": en_metrics,
        "random_forest": rf_metrics
    }

    logger.info("Training complete.")
    return results
