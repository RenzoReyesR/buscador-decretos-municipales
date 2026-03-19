import os
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from transformers import BertTokenizer, BertModel
import torch
from config import RUTA_DOCUMENTOS, INDICE_INVERTIDO_PATH, MONGO_URI, DB_NAME, COLLECTION_NAME, RUTA_EMBEDDINGS, BERT

# Ruta del archivo binario con los embeddings
OUTPUT_PATH = RUTA_EMBEDDINGS

# Inicializar el modelo y tokenizador BETO
tokenizador = BertTokenizer.from_pretrained(BERT)
modelo = BertModel.from_pretrained(BERT)
modelo.eval()

# Función para cargar embeddings existentes
def cargar_embeddings(output_path):
    if os.path.exists(output_path):
        print("[DEBUG] Cargando embeddings existentes.")
        return np.load(output_path, allow_pickle=True).item()
    print("[DEBUG] No se encontraron embeddings previos. Creando nuevo diccionario.")
    return {}

# Función para guardar embeddings actualizados
def guardar_embeddings(output_path, embeddings_dict):
    print(f"[DEBUG] Guardando embeddings en {output_path}")
    np.save(output_path, embeddings_dict)

# Función para extraer texto de un PDF
def pdf_to_text(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=100)
        text = ''
        for image in images:
            text += pytesseract.image_to_string(image)
        print(f"[DEBUG] Texto extraído de {pdf_path} con éxito.")
        return text
    except Exception as e:
        print(f"[ERROR] Error al convertir PDF a texto: {e}")
        return ''

# Función para obtener embeddings de texto
def obtener_embeddings(texto, modelo, tokenizador):
    try:
        inputs = tokenizador(texto, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = modelo(**inputs)
        print("[DEBUG] Embedding generado exitosamente.")
        return outputs.last_hidden_state.mean(dim=1).numpy()
    except Exception as e:
        print(f"[ERROR] Error al generar embeddings: {e}")
        return None

# Función principal para actualizar embeddings
def actualizar_embeddings(pdf_path, filename, output_path):
    print(f"[DEBUG] Iniciando actualización de embeddings para {filename}")
    embeddings_dict = cargar_embeddings(output_path)

    # Verificar si el embedding ya existe
    if filename in embeddings_dict:
        print(f"[INFO] El embedding para {filename} ya existe. No se realizará ninguna actualización.")
        return

    # Extraer texto del PDF
    text = pdf_to_text(pdf_path)

    if text.strip():
        embedding = obtener_embeddings(text, modelo, tokenizador)
        if embedding is not None:
            embeddings_dict[filename] = embedding
            guardar_embeddings(output_path, embeddings_dict)
            print(f"[INFO] Embeddings actualizados y guardados para {filename}")
        else:
            print(f"[ERROR] No se generó el embedding para {filename} debido a un error.")
    else:
        print(f"[WARNING] No se pudo extraer texto del archivo {filename}. No se generó el embedding.")

# Prueba del módulo
def main():
    # Ruta de ejemplo
    pdf_path = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\ejemplo.pdf'
    filename = os.path.basename(pdf_path)

    # Actualizar embeddings
    actualizar_embeddings(pdf_path, filename, OUTPUT_PATH)

if __name__ == "__main__":
    main()
