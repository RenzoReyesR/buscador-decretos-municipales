import json
import os
import re
from nltk.corpus import stopwords
from config_db import collection  # Importar colección desde config_db
from config import RUTA_DOCUMENTOS, INDICE_INVERTIDO_PATH, MONGO_URI, DB_NAME, COLLECTION_NAME, RUTA_EMBEDDINGS, BERT

# Ruta del archivo JSON del índice invertido
JSON_PATH = INDICE_INVERTIDO_PATH
stop_words = set(stopwords.words('spanish'))

# Función para cargar el índice invertido desde un archivo JSON
def cargar_indice(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    return {}

# Función para guardar el índice invertido en un archivo JSON
def guardar_indice(json_path, indice_invertido):
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(indice_invertido, file, ensure_ascii=False, indent=4)

# Función para preprocesar texto y filtrar palabras
def preprocesar_palabras(words):
    processed_words = []
    for word in words:
        word = word.lower()  # Convertir a minúsculas
        word = re.sub(r'[^a-záéíóúñ]', '', word)  # Eliminar caracteres especiales
        if word not in stop_words and len(word) > 2:  # Filtrar stopwords y palabras cortas
            processed_words.append(word)
    return processed_words

# Función para extraer información de norma y año
def extract_norma_number_and_year(filename):
    parts = filename.split('_')
    norma_number = parts[2]
    year = parts[-1].split('.')[0]
    return norma_number, year

# Función para actualizar el índice invertido
def actualizar_indice_invertido(filename, words, json_path):
    indice_invertido = cargar_indice(json_path)

    # Preprocesar las palabras antes de actualizarlas
    processed_words = preprocesar_palabras(words)

    for word in processed_words:
        if word not in indice_invertido:
            indice_invertido[word] = []

        if not any(doc.get('documento') == filename for doc in indice_invertido[word]):
            norma_number, year = extract_norma_number_and_year(filename)
            doc_entry = {
                "documento": filename,
                "numero_norma": norma_number,
                "tf": processed_words.count(word) / len(processed_words),
                "fecha": year,
                "estado": "activo"
            }
            indice_invertido[word].append(doc_entry)

            # Actualizar MongoDB
            collection.update_one(
                {"word": word},
                {"$addToSet": {"documents": doc_entry}},
                upsert=True
            )

    guardar_indice(json_path, indice_invertido)
    print(f"Índice invertido actualizado para {filename}")

# Función para actualizar MongoDB
def actualizar_mongo(words, filename):
    processed_words = preprocesar_palabras(words)
    for word in processed_words:
        norma_number, year = extract_norma_number_and_year(filename)
        doc_entry = {
            "documento": filename,
            "numero_norma": norma_number,
            "tf": processed_words.count(word) / len(processed_words),
            "fecha": year,
            "estado": "activo"
        }
        collection.update_one(
            {"word": word},
            {"$addToSet": {"documents": doc_entry}},
            upsert=True
        )
    print(f"MongoDB actualizado para {filename}")
