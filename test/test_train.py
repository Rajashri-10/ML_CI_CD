import os
import joblib
from src.train import train_model

def test_model_training():
    train_model()
    assert os.path.exists("models/models.pkl")



    model = joblib.load("models/models.pkl")
    assert hasattr(model,"predict")