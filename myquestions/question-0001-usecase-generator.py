import numpy as np
from sklearn.decomposition import PCA
import random

def generar_caso_de_uso_calcular_componentes_pca():
    """
    Genera un caso aleatorio para la función calcular_componentes_pca
    """

    # 1. Dimensiones aleatorias
    n_samples = random.randint(20, 80)
    n_features = random.randint(4, 10)

    # 2. Datos correlacionados
    base = np.random.randn(n_samples, 2)
    ruido = np.random.randn(n_samples, n_features - 2) * 0.2
    X = np.hstack([base @ np.random.randn(2, n_features - 2), ruido])

    # 3. Varianza mínima aleatoria
    varianza_minima = random.uniform(0.6, 0.95)

    input_data = {
        "X": X.copy(),
        "varianza_minima": varianza_minima
    }

    # OUTPUT esperado
    pca = PCA()
    pca.fit(X)
    var_acum = np.cumsum(pca.explained_variance_ratio_)
    n_componentes = np.searchsorted(var_acum, varianza_minima) + 1

    output_data = n_componentes

    return input_data, output_data
