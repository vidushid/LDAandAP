def test_pipeline_runs():
    from src.pipelines.train_pipeline import run_training_pipeline
    results = run_training_pipeline()
    assert "elasticnet" in results
    assert "random_forest" in resultsx
