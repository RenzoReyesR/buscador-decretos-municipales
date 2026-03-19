import os
import time
import threading
import pytesseract
from pdf2image import convert_from_path
import json
import fitz 
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from threading import Lock
from nltk.corpus import stopwords

# Inicializar el bloqueo (lock) para evitar condiciones de carrera
lock = Lock()

# Stopwords en español
stop_words = set(stopwords.words('spanish'))

# Lista en memoria para registrar archivos que están siendo procesados
archivos_en_proceso = set()

# Rutas de archivos y directorios
RUTA_DOCUMENTOS = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\decretos_2023'
OUTPUT_PATH_EMBEDDINGS = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\embeddings_avanzado.npy'
OUTPUT_PATH_INVERTED_INDEX = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\indice_invertido_con_stopwords_normalizados_avanzado.json'
PROCESSED_FILES_PATH = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\archivos_procesados.txt'

# Cargar embeddings previamente guardados
if os.path.exists(OUTPUT_PATH_EMBEDDINGS):
    embeddings_dict = np.load(OUTPUT_PATH_EMBEDDINGS, allow_pickle=True).item()
else:
    embeddings_dict = {}

# Función para convertir PDF a texto
def pdf_to_text(pdf_path):
    """Extrae texto de un PDF usando PyMuPDF (fitz) en lugar de pdf2image y Tesseract."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text

# Normalizar las palabras y eliminar caracteres especiales
def normalizar_palabra(palabra):
    palabra = re.sub(r'[^\w\s]', '', palabra)
    palabra = palabra.lstrip('0')
    palabra = palabra.strip('_')
    palabra = palabra.strip(':')
    palabra = palabra.strip(';')
    return palabra

# Función para procesar el texto extraído, aplicando normalización y eliminación de stopwords
def process_text(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Eliminar stopwords
    filtered_words = [normalizar_palabra(word) for word in words if word not in stop_words]
    
    return filtered_words

# Función para cargar el índice invertido desde un archivo JSON
def load_inverted_index():
    if os.path.exists(OUTPUT_PATH_INVERTED_INDEX):
        try:
            with open(OUTPUT_PATH_INVERTED_INDEX, 'r', encoding='utf-8') as f:
                inverted_index = json.load(f)
                print(f"Índice invertido cargado con éxito: {len(inverted_index)} términos")  # Depuración
                return inverted_index
        except json.JSONDecodeError as e:
            print(f"Error al cargar el índice invertido: {e}")
            return {}
    else:
        print(f"No se encontró el archivo de índice invertido en: {OUTPUT_PATH_INVERTED_INDEX}")
    return {}

# Función para guardar el índice invertido en un archivo JSON
def save_inverted_index_to_json(inverted_index):
    try:
        print(f"Guardando índice invertido en: {OUTPUT_PATH_INVERTED_INDEX}")  # Mensaje de depuración
        with open(OUTPUT_PATH_INVERTED_INDEX, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f, ensure_ascii=False, indent=4)
        print(f"Índice invertido guardado con éxito: {len(inverted_index)} términos")  # Depuración
    except IOError as e:
        print(f"Error al guardar el índice invertido: {e}")

# Función para actualizar el índice invertido (sin duplicar)
def update_inverted_index(inverted_index, pdf_path, filename):
    text = pdf_to_text(pdf_path)
    
    # Aplicar procesamiento (normalización y stopwords)
    filtered_words = process_text(text)
    
    for word in filtered_words:
        if word not in inverted_index:
            inverted_index[word] = []
        if filename not in inverted_index[word]:
            inverted_index[word].append(filename)
    
    print(f"Índice invertido actualizado con {filename}. Total de términos: {len(inverted_index)}")  # Mensaje de depuración
    return inverted_index

# Función para obtener embeddings de texto utilizando BERT
def obtener_embeddings(texto, modelo, tokenizador):
    inputs = tokenizador(texto, return_tensors='pt', truncation=True, padding=True)
    outputs = modelo(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Función para registrar archivo procesado en un archivo de texto (con bloqueo)
def registrar_archivo_procesado(filename):
        try:
            print(f"Registrando archivo procesado: {filename}")  # Depuración
            with open(PROCESSED_FILES_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{filename}\n")
            print(f"Archivo registrado con éxito: {filename}")  # Depuración
        except IOError as e:
            print(f"Error al registrar el archivo procesado: {e}")

# Función para cargar la lista de archivos ya procesados (con bloqueo)
def cargar_archivos_procesados():
    with lock:
        if os.path.exists(PROCESSED_FILES_PATH):
            try:
                with open(PROCESSED_FILES_PATH, 'r', encoding='utf-8') as f:
                    processed_files = set(f.read().splitlines())
                    print(f"Archivos procesados cargados con éxito: {len(processed_files)} archivos")  # Depuración
                    return processed_files
            except IOError as e:
                print(f"Error al cargar archivos procesados: {e}")
                return set()
        return set()

# Función para procesar PDF y actualizar embeddings e índice
def process_pdf(pdf_path, filename, inverted_index, tokenizador, modelo, collection):
    print(f'Procesando archivo: {filename}')

    # Asegurar que el archivo no esté ya en proceso
    with lock:
        if filename in archivos_en_proceso:
            print(f"Archivo {filename} ya está en proceso. Omitiendo.")
            return
        archivos_en_proceso.add(filename)

    # Verificar si ya ha sido procesado
    archivos_procesados = cargar_archivos_procesados()
    if filename in archivos_procesados:
        print(f"Archivo {filename} ya fue procesado anteriormente. Omitiendo.")
        archivos_en_proceso.remove(filename)
        return

    # Verificar si el archivo ya está en el índice invertido
    if any(filename in docs for docs in inverted_index.values()):
        print(f"Archivo {filename} ya está en el índice invertido. Omitiendo.")
        archivos_en_proceso.remove(filename)
        return

    # Verificar si el archivo ya tiene embeddings calculados
    if filename in embeddings_dict:
        print(f"Embeddings ya existen para el archivo: {filename}. Omitiendo.")
        archivos_en_proceso.remove(filename)
        return

    # Actualizar índice invertido (sin duplicar)
    inverted_index = update_inverted_index(inverted_index, pdf_path, filename)
    
    # Obtener texto del PDF y calcular embeddings
    text = pdf_to_text(pdf_path)
    if text.strip():  # Asegurarse de que no esté vacío
        embedding = obtener_embeddings(text, modelo, tokenizador)
        embeddings_dict[filename] = embedding
        print(f"Embeddings calculados para el archivo: {filename}")
    else:
        print(f"Advertencia: No se pudo extraer texto del archivo {filename}")

    # Guardar embeddings actualizados
    print(f"Guardando embeddings en: {OUTPUT_PATH_EMBEDDINGS}")
    np.save(OUTPUT_PATH_EMBEDDINGS, embeddings_dict)

    # Guardar archivo como procesado (uso del lock para evitar conflictos entre hilos)
    registrar_archivo_procesado(filename)

    # Guardar el índice invertido actualizado en el archivo JSON
    save_inverted_index_to_json(inverted_index)

    # Remover el archivo de la lista de archivos en proceso
    with lock:
        archivos_en_proceso.remove(filename)

# Revisar si hay nuevos archivos para procesar
def check_for_new_files(folder_path, inverted_index, tokenizador, modelo, collection):
    current_files = set(os.listdir(folder_path))
    processed_files = cargar_archivos_procesados()  # Archivos ya procesados
    
    # Archivos nuevos que no están en el archivo de texto
    new_files = current_files - processed_files
    
    new_files_count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for filename in new_files:
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(folder_path, filename)
                new_files_count += 1
                executor.submit(process_pdf, pdf_path, filename, inverted_index, tokenizador, modelo, collection)
    
    if new_files_count > 0:
        print(f'{new_files_count} nuevos archivos detectados y procesados.')
    else:
        print('No se detectaron nuevos archivos.')

# Función principal para ejecutar el daemon
def run(folder_path, tokenizador, modelo, collection):
    inverted_index = load_inverted_index()  # Cargar el índice invertido desde el archivo JSON
    while True:
        check_for_new_files(folder_path, inverted_index, tokenizador, modelo, collection)
        time.sleep(60)  # Esperar 60 segundos antes de verificar nuevamente

# Función para iniciar el daemon
def start_daemon(folder_path, tokenizador, modelo, collection):
    thread = threading.Thread(target=run, args=(folder_path, tokenizador, modelo, collection), daemon=True)
    thread.start()
