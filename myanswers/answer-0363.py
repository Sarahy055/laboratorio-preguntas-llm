import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalizar_min_max(df, columnas):
    """
    Selecciona las columnas indicadas, aplica MinMaxScaler para transformarlas
    al rango [0, 1] reemplazando los valores originales, y devuelve el DataFrame modificado.
    
    Argumentos:
    df (pandas.DataFrame): El DataFrame original con los datos.
    columnas (list): Lista de strings con los nombres de las columnas a normalizar.
    
    Devuelve:
    pandas.DataFrame: El DataFrame con las columnas especificadas normalizadas.
    """
    # 1. Hacemos una copia para evitar advertencias de SettingWithCopyWarning y trabajar seguros
    df_resultado = df.copy()
    
    # 2. Inicializamos el MinMaxScaler de sklearn
    scaler = MinMaxScaler()
    
    # 3. Ajustamos y transformamos las columnas seleccionadas, reemplazando sus valores
    df_resultado[columnas] = scaler.fit_transform(df_resultado[columnas])
    
    # 4. Devolvemos el DataFrame modificado
    return df_resultado
