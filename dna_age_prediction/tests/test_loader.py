from src.data.loader import DataLoader

def test_loader_shapes():
    X, y = DataLoader.create_sample_data(50, 100)
    assert X.shape == (50, 100)
    assert len(y) == 50
