import pandas as pd
from sklearn.linear_model import LinearRegression

def predecir_consumo(X, y, nueva_instancia):
    """
    Entrena un modelo de Regresión Lineal y predice el consumo para una nueva instancia.
    
    Argumentos:
    X (pandas.DataFrame): DataFrame con columnas 'temperatura' y 'habitantes'.
    y (numpy.ndarray): Array con el consumo eléctrico real.
    nueva_instancia (dict): Diccionario con los datos del nuevo hogar.
    
    Devuelve:
    float: El consumo eléctrico predicho.
    """
    # 1. Inicializar y entrenar el modelo de Regresión Lineal
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # 2. Convertir el diccionario de la nueva instancia en un DataFrame de una sola fila
    # Pasamos el diccionario dentro de una lista para que pandas reconozca las llaves como columnas
    X_nuevo = pd.DataFrame([nueva_instancia])
    
    # 3. Realizar la predicción, extraer el primer elemento y asegurar que sea un float de Python
    prediccion = float(modelo.predict(X_nuevo)[0])
    
    return prediccion
