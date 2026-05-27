import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def estimar_covarianza_regularizada(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Calcula la matriz de covarianza regularizada usando el estimador Ledoit-Wolf
    sobre las columnas seleccionadas y limpias de nulos.
    
    Argumentos:
    df (pd.DataFrame): DataFrame original con datos de monitoreo ambiental.
    feature_cols (list[str]): Nombres de las columnas numéricas correlacionadas a evaluar.
    
    Devuelve:
    pd.DataFrame: Matriz cuadrada de covarianza estimada con índices y columnas correspondientes.
    """
    # 1. Seleccionar únicamente las columnas indicadas y eliminar filas con al menos un valor nulo
    # Filtramos primero las columnas y aplicamos dropna sobre el eje de las filas (axis=0)
    cleaned_df = df[feature_cols].dropna(axis=0, how="any")
    
    # 2. Convertir el bloque limpio a un arreglo de numpy de tipo flotante
    feature_matrix = cleaned_df.to_numpy(dtype=float)
    
    # 3. Ajustar un modelo LedoitWolf sobre la matriz numérica y obtener la covarianza estimada
    covariance_model = LedoitWolf()
    covariance_model.fit(feature_matrix)
    covariance_matrix = covariance_model.covariance_
    
    # 4. Devolver un DataFrame cuadrado cuyas filas y columnas correspondan exactamente a feature_cols
    expected_output = pd.DataFrame(
        covariance_matrix,
        index=feature_cols,
        columns=feature_cols
    )
    
    return expected_output
