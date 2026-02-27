import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_normalizar_por_grupo():
    """
    Genera caso de uso para normalizar_por_grupo
    """

    n_rows = random.randint(20, 60)
    n_features = random.randint(2, 5)
    n_grupos = random.randint(2, 4)

    data = np.random.randn(n_rows, n_features)
    grupos = np.random.choice([f"G{i}" for i in range(n_grupos)], size=n_rows)

    cols = [f"x{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=cols)
    group_col = "grupo"
    df[group_col] = grupos

    input_data = {
        "df": df.copy(),
        "group_col": group_col
    }

    numeric = df.drop(columns=[group_col])
    grouped = df.groupby(group_col)

    norm = grouped[numeric.columns].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )

    output_data = norm.to_numpy()

    return input_data, output_data
