import os
import numpy as np
import pytesseract
import time
from pdf2image import convert_from_path
from transformers import DistilBertTokenizer, DistilBertModel
from concurrent.futures import ThreadPoolExecutor
import torch

# Ruta de la carpeta donde se encuentran los documentos PDF
RUTA_DOCUMENTOS = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\decretos_2023_test'
OUTPUT_PATH = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\embeddings_avanzado_test.npy'

# Inicializar DistilBERT multilingüe
tokenizador = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
modelo = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
modelo.eval()  # Colocar el modelo en modo de evaluación

# Verificar si hay una GPU disponible y mover el modelo a la GPU si es posible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)

# Función para obtener embeddings de DistilBERT
def obtener_embeddings(texto, modelo, tokenizador):
    inputs = tokenizador(texto, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = modelo(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()  # Mover a la CPU y convertir a numpy
    return np.squeeze(embedding)  # Asegurarse de que sea un vector unidimensional

# Función para extraer texto de un PDF
def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path, dpi=100)  # Reducir DPI para mejorar la velocidad
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# Función para procesar un archivo PDF y calcular embeddings
def procesar_pdf(pdf_path):
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        text = pdf_to_text(pdf_path)
        if text.strip():  # Verifica que el texto no esté vacío
            embedding = obtener_embeddings(text, modelo, tokenizador)
            if embedding.shape == (768,):  # Verificar que sea un vector unidimensional
                return doc_id, embedding
            else:
                print(f"Advertencia: El embedding de {doc_id} tiene dimensiones inesperadas: {embedding.shape}")
                return None
        else:
            print(f"Advertencia: No se pudo extraer texto del archivo {pdf_path}")
            return None
    except Exception as e:
        print(f"Error al procesar {pdf_path}: {e}")
        return None

# Calcular y guardar los embeddings usando procesamiento en paralelo
start_time = time.time()  # Iniciar medición de tiempo
embeddings_dict = {}

with ThreadPoolExecutor() as executor:
    pdf_files = [os.path.join(RUTA_DOCUMENTOS, f) for f in os.listdir(RUTA_DOCUMENTOS) if f.endswith('.pdf')]
    results = executor.map(procesar_pdf, pdf_files)

    for result in results:
        if result:
            doc_id, embedding = result
            embeddings_dict[doc_id] = embedding
            print(f"Embedding generado para {doc_id}")

# Guardar los embeddings en un archivo .npy
np.save(OUTPUT_PATH, embeddings_dict)
print(f"Embeddings guardados en {OUTPUT_PATH}")

end_time = time.time()  # Finalizar medición de tiempo
elapsed_time = end_time - start_time  # Calcular tiempo total

# Mostrar el tiempo en formato horas:minutos:segundos y en segundos
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Tiempo total de generación de embeddings: {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss) o {elapsed_time:.2f} segundos")
