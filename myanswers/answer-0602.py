import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def transformar_datos_oceanicos(df, target_col):
    """
    Separa características de la variable objetivo, imputa nulos con la media,
    aplica transformación logarítmica log(1+x), escala a [0, 1] y devuelve arrays de numpy.
    
    Argumentos:
    df (pandas.DataFrame): El DataFrame con los datos oceanográficos.
    target_col (str): El nombre de la columna que actúa como variable objetivo.
    
    Devuelve:
    tuple: (X_escalada, y) donde ambos son arrays de NumPy.
    """
    # 1. Separar características X de la variable objetivo y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()
    
    # 2. Imputar valores faltantes usando el promedio de cada columna
    imputer = SimpleImputer(strategy="mean")
    X_imputada = imputer.fit_transform(X)
    
    # 3. Aplicar transformación logarítmica log(1 + x)
    # np.log1p calcula de forma precisa log(1 + x) elemento por elemento
    X_log = np.log1p(X_imputada)
    
    # 4. Escalar las características en el rango [0, 1]
    scaler = MinMaxScaler()
    X_escalada = scaler.fit_transform(X_log)
    
    # 5. Devolver la matriz X procesada y el vector y
    return X_escalada, y
