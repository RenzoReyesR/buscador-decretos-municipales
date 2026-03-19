from flask import Flask, request, render_template, send_file
from facade import BuscadorFacade
import os
import threading
import time
from config import RUTA_DOCUMENTOS, INDICE_INVERTIDO_PATH,RUTA_EMBEDDINGS, BERT,  MONGO_URI, DB_NAME, COLLECTION_NAME

app = Flask(__name__)

# Variables globales
facade = None  # Inicialmente vacío
estado_sistema = {"flask": "OK", "mongodb": "Cargando", "bert": "Cargando", "crawler": "Cargando"}
tiempos_carga = {}

# Función para inicializar la fachada
def inicializar_facade():
    global facade
    global estado_sistema
    start_time = time.time()
    try:
        facade = BuscadorFacade(RUTA_DOCUMENTOS, debug=True)
        facade.inicializar_bert()  # Inicialización diferida de BERT
        estado_sistema["bert"] = "OK"
        tiempos_carga["bert"] = round(time.time() - start_time, 2)
        print(f"[DEBUG] BERT cargado en {tiempos_carga['bert']} segundos.")
    except Exception as e:
        estado_sistema["bert"] = f"Error: {e}"
        print(f"[ERROR] Error al cargar BERT: {e}")

# Función para inicializar MongoDB
def inicializar_mongodb():
    global estado_sistema
    start_time = time.time()
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        estado_sistema["mongodb"] = "OK"
        tiempos_carga["mongodb"] = round(time.time() - start_time, 2)
        print(f"[DEBUG] MongoDB cargado en {tiempos_carga['mongodb']} segundos.")
    except Exception as e:
        estado_sistema["mongodb"] = f"Error: {e}"
        print(f"[ERROR] Error al conectar con MongoDB: {e}")

# Función para inicializar el crawler
def inicializar_crawler():
    global estado_sistema
    start_time = time.time()
    try:
        from crawler import start_crawler
        start_crawler(RUTA_DOCUMENTOS,
                      INDICE_INVERTIDO_PATH,
                      RUTA_EMBEDDINGS)
        estado_sistema["crawler"] = "OK"
        tiempos_carga["crawler"] = round(time.time() - start_time, 2)
        print(f"[DEBUG] Crawler inicializado en {tiempos_carga['crawler']} segundos.")
    except Exception as e:
        estado_sistema["crawler"] = f"Error: {e}"
        print(f"[ERROR] Error al inicializar el crawler: {e}")

# Página de inicio
@app.route("/")
def index():
    return render_template("index.html", estado_sistema=estado_sistema, tiempos_carga=tiempos_carga)

@app.route("/buscar", methods=["POST"])
def buscar():
    global facade
    query = request.form.get("query")
    if not query:
        return render_template("error.html", error_message="Por favor, ingrese una consulta.")

    try:
        resultados = facade.buscar_documentos(query)
        if not resultados:
            return render_template("error.html", error_message="No se encontraron documentos para la consulta.")
        return render_template("resultados.html", resultados=resultados)
    except Exception as e:
        return render_template("error.html", error_message=f"Ocurrió un error: {e}")

@app.route("/ver/<filename>")
def ver_archivo(filename):
    """Permite ver un archivo en el navegador."""
    try:
        filepath = os.path.join(RUTA_DOCUMENTOS, filename)
        print(f"[DEBUG] Intentando abrir el archivo: {filepath}")
        if not os.path.exists(filepath):
            print(f"[ERROR] Archivo no encontrado: {filepath}")
            return render_template("error.html", mensaje="El archivo no existe.")
        return send_file(filepath)
    except Exception as e:
        print(f"[ERROR] Error al mostrar el archivo: {e}")
        return render_template("error.html", mensaje="Ocurrió un error al intentar mostrar el archivo.")

@app.route("/descargar/<filename>")
def descargar_archivo(filename):
    """Permite descargar un archivo."""
    try:
        filepath = os.path.join(RUTA_DOCUMENTOS, filename)
        print(f"[DEBUG] Intentando descargar el archivo: {filepath}")
        if not os.path.exists(filepath):
            print(f"[ERROR] Archivo no encontrado: {filepath}")
            return render_template("error.html", mensaje="El archivo no existe.")
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        print(f"[ERROR] Error al descargar el archivo: {e}")
        return render_template("error.html", mensaje="Ocurrió un error al intentar descargar el archivo.")

if __name__ == "__main__":
    # Inicializar Flask
    print("[DEBUG] Iniciando Flask...")
    estado_sistema["flask"] = "OK"

    # Cargar componentes en hilos separados
    threading.Thread(target=inicializar_mongodb, daemon=True).start()
    threading.Thread(target=inicializar_facade, daemon=True).start()
    threading.Thread(target=inicializar_crawler, daemon=True).start()

    app.run(debug=True)
