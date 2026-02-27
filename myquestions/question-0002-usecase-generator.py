import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import random

def generar_caso_de_uso_calcular_auc_logistico():
    """
    Genera un caso aleatorio para calcular_auc_logistico
    """

    n_rows = random.randint(50, 120)
    n_features = random.randint(3, 6)

    X = np.random.randn(n_rows, n_features)
    coef = np.random.randn(n_features)
    logits = X @ coef
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    target_col = "target"
    df[target_col] = y

    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    X_train = df.drop(columns=[target_col])
    y_train = df[target_col]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    prob_pred = model.predict_proba(X_train)[:, 1]

    output_data = roc_auc_score(y_train, prob_pred)

    return input_data, output_data
