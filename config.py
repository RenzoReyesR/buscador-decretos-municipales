import os

# Configuración del directorio base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de archivos y documentos
RUTA_DOCUMENTOS = os.path.join(BASE_DIR, "decretos_2023_test")  # Carpeta con PDFs
INDICE_INVERTIDO_PATH = os.path.join(BASE_DIR, "indice_invertido_con_stopwords_normalizados_avanzado_test.json")  # Índice invertido en JSON
RUTA_EMBEDDINGS = os.path.join(BASE_DIR, "embeddings_avanzado_test.npy")  # Archivo de embeddings

# Configuración de MongoDB
#MONGO_URI, DB_NAME, COLLECTION_NAME
MONGO_URI = "mongodb://localhost:27017/" # Requiere MongoDB corriendo localmente
DB_NAME = "indice_invertido_decretos_munvalp_test"
COLLECTION_NAME = "indice_invertido_test"

# Configuración de Flask
FLASK_DEBUG = True

# Intervalo del Crawler
CHECK_INTERVAL = 240  # Intervalo en segundos para revisar nuevos documentos

BERT = 'dccuchile/bert-base-spanish-wwm-cased'