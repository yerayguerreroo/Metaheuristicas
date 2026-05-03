import joblib
import numpy as np


class BlackBoxModel:

    #Considerar blackbox_modelA y blackbox_modelB
    def __init__(self, path="blackbox_model.pkl"):
        self.model = joblib.load(path)

    def predict(self, x):
        x = np.array(x)
        # Si es un solo punto (array de 1 dimensión, ej: [0.5, -0.2])
        if x.ndim == 1:
            return self.model.predict(x.reshape(1, -1))[0]
        # Si es un batch de puntos (array de 2 dimensiones, ej: [[..], [..]])
        else:
            return self.model.predict(x)