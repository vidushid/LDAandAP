import numpy as np
from src.models.elasticnet import ElasticNetModel

def test_elasticnet_fit_predict():
    X = np.random.rand(50, 20)
    y = np.random.rand(50)

    model = ElasticNetModel()
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == 50
