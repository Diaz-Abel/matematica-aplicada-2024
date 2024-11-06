# Proyecto Final - Matemática Aplicada

## Descripción
Este proyecto realiza un análisis de sentimientos utilizando un dataset de tweets. Implementa un sistema de inferencia difusa para clasificar los sentimientos en positivo, negativo o neutral.

## Requisitos
Para ejecutar este proyecto, asegúrate de tener instalado Python 3.x y las siguientes dependencias:

- pandas
- nltk
- numpy
- scikit-fuzzy
- matplotlib

## Instalación
1. Clona este repositorio.
2. Navega a la carpeta del proyecto.
3. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/Mac
   .\venv\Scripts\activate  # En Windows
4. Instala las dependencias:
    ```bash 
        pip install -r requirements.txt
    ```
### Dataset Necesario
Este proyecto utiliza el dataset de tweets de Kaggle para el análisis de sentimientos. Para obtenerlo:
1. Descarga el dataset desde: [aquí](https://www.kaggle.com/datasets/krishbaisoya/tweets-sentiment-analysis?resource=download)
2. Coloca los archivos .csv en la carpeta raíz del proyecto.

## Uso
Para ejecutar el análisis de sentimientos, utiliza el siguiente comando:
```bash 
python proyecto.py
 ```

## Notas sobre el Lexicón de VADER
El proyecto utiliza el lexicón de VADER para calcular los puntajes de sentimiento. La primera vez que se ejecuta el proyecto, el lexicón se descarga automáticamente si tienes conexión a internet.  
Tras esta descarga inicial, puedes comentar la línea `nltk.download("vader_lexicon")` en el código para evitar descargarlo nuevamente en futuras ejecuciones.