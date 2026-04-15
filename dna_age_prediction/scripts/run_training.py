from src.pipelines.train_pipeline import run_training_pipeline

if __name__ == "__main__":
    results = run_training_pipeline()
    print("Final Results:")
    print(results)
