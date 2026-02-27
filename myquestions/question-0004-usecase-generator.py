import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

def generar_caso_de_uso_matriz_similitud_coseno():
    """
    Genera caso de uso para matriz_similitud_coseno
    """

    n_samples = random.randint(5, 15)
    n_features = random.randint(3, 7)

    X = np.random.randn(n_samples, n_features)

    input_data = {
        "X": X.copy()
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sim = cosine_similarity(X_scaled)

    output_data = sim

    return input_data, output_data
