import os
import sys
import pandas as pd
import re
import nltk
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

# Módulo 1: Creación y Preprocesamiento del Dataset

# Cargar el dataset desde un archivo CSV
url = "test_data.csv"
# url = "train_data.csv"

# Verificar si el archivo existe
if os.path.isfile(url):
    # Cargar el dataset desde el archivo CSV
    df = pd.read_csv(url)
    print("Archivo cargado exitosamente.")
else:
    print(f"Error: No se encontró el archivo en la ruta '{url}'")
    sys.exit("Terminando el programa debido a un archivo faltante.")

# Renombrar columnas para mayor claridad
df.rename(columns={"sentence": "oracion", "sentiment": "label_original"}, inplace=True)


# Función para preprocesar el texto
def preprocesar(oracion):
    """
    Preprocesa una oración eliminando caracteres no alfabéticos y palabras de una sola letra.

    Parámetros:
    oracion (str): Cadena de texto de la oración a procesar.

    Retorna:
    str: La oración preprocesada sin caracteres no alfabéticos y sin palabras de una sola letra.
    """
    oracion = re.sub(r"[^a-zA-Z\s]", "", oracion)
    oracion = " ".join([palabra for palabra in oracion.split() if len(palabra) > 1])
    return oracion


# Preprocesar las oraciones y almacenarlas en una nueva columna
for indice, fila in df.iterrows():
    df.at[indice, "oracion_procesada"] = preprocesar(fila["oracion"])

# Módulo 2: Lexicón de Sentimientos y Cálculo de Puntajes

# Agregar las columnas que se van a necesitar
df[
    [
        "puntaje_positivo",
        "puntaje_negativo",
        "puntaje_neutro",
        "deffuz",
        "label_calculado",
        "tiempos",
    ]
] = None

# Descargar el lexicón de VADER para realizar el análisis de sentimientos
# nltk.download("vader_lexicon")

# Inicializar el analizador de sentimientos
sia = SentimentIntensityAnalyzer()


# Función para calcular los puntajes de sentimientos
def calcular_puntajes(oracion):
    """
    Calcula los puntajes de sentimiento (positivo, negativo y neutro) de una oración utilizando el lexicón de VADER.

    Parámetros:
    oracion (str): Oración procesada para analizar.

    Retorna:
    pd.Series: Serie con los puntajes positivo, negativo y neutro.
    """
    puntajes = sia.polarity_scores(oracion)

    return [puntajes["pos"], puntajes["neg"], puntajes["neu"]]


# Módulo 3: Fuzzificación

# Iterar sobre cada fila del DataFrame para calcular los puntajes
for indice, fila in df.iterrows():
    start_time = time.perf_counter()

    puntajes = calcular_puntajes(fila["oracion_procesada"])  # Calcular puntajes

    execution_time = time.perf_counter() - start_time  # Calcular el tiempo transcurrido

    df.at[indice, "puntaje_positivo"] = puntajes[0]
    df.at[indice, "puntaje_negativo"] = puntajes[1]
    df.at[indice, "puntaje_neutro"] = puntajes[2]
    df.at[indice, "tiempos"] = execution_time  # Almacenar el tiempo

# Calcular los mínimos, máximos y el punto medio de las puntuaciones
p_min = df["puntaje_positivo"].min()
p_max = df["puntaje_positivo"].max()
p_mid = (p_max + p_min) / 2

n_min = df["puntaje_negativo"].min()
n_max = df["puntaje_negativo"].max()
n_mid = (n_max + n_min) / 2

# Definir las variables del universo
puntaje_positivo = ctrl.Antecedent(
    np.arange(p_min, p_max + 0.1, 0.1), "puntaje_positivo"
)
puntaje_negativo = ctrl.Antecedent(
    np.arange(n_min, n_max + 0.1, 0.1), "puntaje_negativo"
)

resultado = ctrl.Consequent(np.arange(0, 11, 1), "resultado")

# Funciones de pertenencia para puntajes positivos
puntaje_positivo["bajo"] = fuzz.trimf(puntaje_positivo.universe, [p_min, p_min, p_mid])
puntaje_positivo["medio"] = fuzz.trimf(puntaje_positivo.universe, [p_min, p_mid, p_max])
puntaje_positivo["alto"] = fuzz.trimf(puntaje_positivo.universe, [p_mid, p_max, p_max])

# Funciones de pertenencia para puntajes negativos
puntaje_negativo["bajo"] = fuzz.trimf(puntaje_negativo.universe, [n_min, n_min, n_mid])
puntaje_negativo["medio"] = fuzz.trimf(puntaje_negativo.universe, [n_min, n_mid, n_max])
puntaje_negativo["alto"] = fuzz.trimf(puntaje_negativo.universe, [n_mid, n_max, n_max])

# Funciones de pertenencia para la salida
resultado["negativo"] = fuzz.trimf(resultado.universe, [0, 0, 5])
resultado["neutral"] = fuzz.trimf(resultado.universe, [0, 5, 10])
resultado["positivo"] = fuzz.trimf(resultado.universe, [5, 10, 10])

# Módulo 4: Base de reglas

# Definir las reglas
regla1 = ctrl.Rule(
    puntaje_positivo["bajo"] & puntaje_negativo["bajo"], resultado["neutral"]
)
regla2 = ctrl.Rule(
    puntaje_positivo["medio"] & puntaje_negativo["bajo"], resultado["positivo"]
)
regla3 = ctrl.Rule(
    puntaje_positivo["alto"] & puntaje_negativo["bajo"], resultado["positivo"]
)
regla4 = ctrl.Rule(
    puntaje_positivo["bajo"] & puntaje_negativo["medio"], resultado["negativo"]
)
regla5 = ctrl.Rule(
    puntaje_positivo["medio"] & puntaje_negativo["medio"], resultado["neutral"]
)
regla6 = ctrl.Rule(
    puntaje_positivo["alto"] & puntaje_negativo["medio"], resultado["positivo"]
)
regla7 = ctrl.Rule(
    puntaje_positivo["bajo"] & puntaje_negativo["alto"], resultado["negativo"]
)
regla8 = ctrl.Rule(
    puntaje_positivo["medio"] & puntaje_negativo["alto"], resultado["negativo"]
)
regla9 = ctrl.Rule(
    puntaje_positivo["alto"] & puntaje_negativo["alto"], resultado["neutral"]
)

# Crear el sistema de control
sistema_control = ctrl.ControlSystem(
    [regla1, regla2, regla3, regla4, regla5, regla6, regla7, regla8, regla9]
)
simulador = ctrl.ControlSystemSimulation(sistema_control)

# Módulo 5: Defuzzificación

# Aplicar las reglas de inferencia difusa para cada oración
for indice, fila in df.iterrows():
    simulador.input["puntaje_positivo"] = fila["puntaje_positivo"]
    simulador.input["puntaje_negativo"] = fila["puntaje_negativo"]
    # Se aplica las reglas,
    simulador.compute()

    op = simulador.output["resultado"]
    df.at[indice, "deffuz"] = op

    # Clasificar el resultado
    if op < 3.3:
        df.at[indice, "label_calculado"] = "Negativo"
    elif 3.3 <= op < 6.7:
        df.at[indice, "label_calculado"] = "Neutral"
    elif 6.7 <= op:
        df.at[indice, "label_calculado"] = "Positivo"

# Imprimir la cantidad de clasificaciones
print("Cantidad de tweets Positivo:", df[df["label_calculado"] == "Positivo"].shape[0])
print("Cantidad de tweets Negativo:", df[df["label_calculado"] == "Negativo"].shape[0])
print("Cantidad de tweets Neutral:", df[df["label_calculado"] == "Neutral"].shape[0])

tiempo_promedio = df["tiempos"].mean()
print("Tiempo de ejecucion promedio:", tiempo_promedio)

# Renombrar columnas para mejor claridad
df.rename(
    columns={
        "oracion": "Oracion original",
        "label_original": "Label original",
        "puntaje_positivo": "Puntaje positivo",
        "puntaje_negativo": "Puntaje negativo",
        "puntaje_neutro": "Puntaje neutral",
        "deffuz": "Resultado de inferencia",
        "tiempos": "Tiempo de ejecucion",
    },
    inplace=True,
)

# Módulo 6: Benchmarks

df_nuevo = df[
    [
        "Oracion original",
        "Label original",
        "Puntaje positivo",
        "Puntaje negativo",
        "Puntaje neutral",
        "Resultado de inferencia",
        "Tiempo de ejecucion",
    ]
].copy()

# Determinar el nombre del archivo basado en la URL
if "test" in url:
    nombre_archivo = "proyecto_test.csv"
elif "train" in url:
    nombre_archivo = "proyecto_train.csv"

# Guardar el DataFrame procesado en un archivo CSV
df_nuevo.to_csv(nombre_archivo, index=False)
