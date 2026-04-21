import joblib
import numpy as np


class BlackBoxModel:

    #Considerar blackbox_modelA y blackbox_modelB
    def __init__(self, path="blackbox_model.pkl"):
        self.model = joblib.load(path)

    def predict(self, x):
        x = np.array(x).reshape(1, -1)
        return self.model.predict(x)[0]