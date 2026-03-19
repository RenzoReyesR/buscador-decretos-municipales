import os
import time
import numpy as np
import torch
from pymongo import MongoClient
from transformers import BertTokenizer, BertModel
from actualizar_embeddings import cargar_embeddings, actualizar_embeddings
from actualizar_indice_invertido import actualizar_indice_invertido
from crawler import check_for_new_files
from procesar_consulta import run
from config import RUTA_DOCUMENTOS, INDICE_INVERTIDO_PATH, MONGO_URI, DB_NAME, COLLECTION_NAME, BERT, RUTA_EMBEDDINGS

class BuscadorFacade:
    def __init__(self, ruta_documentos: str, debug: bool = False):
        self.ruta_documentos = ruta_documentos
        self.debug = debug
        self.RUTA_DOCUMENTOS = RUTA_DOCUMENTOS
        self.INDICE_INVERTIDO_PATH = INDICE_INVERTIDO_PATH
        self.embeddings_path = RUTA_EMBEDDINGS
        # Conexión a MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        self.collection = db[COLLECTION_NAME]

        # Inicializar embeddings
        self.embeddings = cargar_embeddings(self.embeddings_path)

        # Inicialización diferida de distilBERT
        self.tokenizador = None
        self.modelo = None

    def inicializar_bert(self):
        """Inicializar BERT de forma diferida."""
        try:
            print("[INFO] Inicializando BERT...")
            start_time = time.time()
            self.tokenizador = BertTokenizer.from_pretrained(BERT)
            self.modelo = BertModel.from_pretrained(BERT)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.modelo.to(device)
            self.modelo.eval()  # Modo evaluación
            print(f"[INFO] BERT inicializado en {round(time.time() - start_time, 2)} segundos.")
        except Exception as e:
            print(f"[ERROR] Error al inicializar BERT: {e}")

    def buscar_documentos(self, query: str):
        """Buscar documentos utilizando el índice invertido y los embeddings."""
        if not self.modelo or not self.tokenizador:
            raise RuntimeError("BERT no está inicializado.")
        # Procesamiento de consulta y búsqueda
        # Aquí incluirías las llamadas necesarias a tus funciones específicas
        return run(query)

    def obtener_embeddings(self, texto: str):
        """Obtener embeddings para un texto usando BERT."""
        inputs = self.tokenizador(texto, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.modelo(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy().squeeze()

    def actualizar_indice(self, pdf_path: str):
        """Actualizar el índice invertido."""
        filename = os.path.basename(pdf_path)
        actualizar_indice_invertido(pdf_path, filename, self.output_path, self.collection)

    def actualizar_embeddings(self, pdf_path: str):
        """Actualizar embeddings."""
        filename = os.path.basename(pdf_path)
        actualizar_embeddings(pdf_path, filename, self.embeddings_path)

    def ejecutar_crawler(self):
        """Ejecutar el crawler para verificar nuevos archivos."""
        check_for_new_files(self.ruta_documentos, self.output_path, self.embeddings_path)
