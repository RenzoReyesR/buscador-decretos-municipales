import os
import json
import re
import pytesseract
import time
from pdf2image import convert_from_path
from PIL import Image
from nltk.corpus import stopwords
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import log
from pymongo.errors import ConnectionFailure
import matplotlib.pyplot as plt
from collections import Counter

# Cargar stopwords en español y agregar letras individuales y números a eliminar
stop_words = set(stopwords.words('spanish'))
stop_words.update(list("abcdefghijklmnopqrstuvwxyz"))
stop_words.update([str(i) for i in range(10)])  # Filtrar números

# Funciones auxiliares

def plot_histogram(data, title, xlabel, ylabel, output_path, top_n):
    """Generar histograma con los primeros N datos."""
    terms, frequencies = zip(*data[:top_n])
    plt.figure(figsize=(12, 6))
    plt.bar(terms, frequencies)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_pdf_file(pdf_path):
    """Procesar un archivo PDF y devolver las palabras utilizadas, eliminadas y el número de páginas."""
    text = ""
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            text += pytesseract.image_to_string(image)
        text = text.lower()
        words = re.findall(r'\b[a-záéíóúñü]+\b', text)  # Solo palabras, sin números ni símbolos
        used_words = [word for word in words if word not in stop_words]
        removed_words = [word for word in words if word in stop_words]
        return used_words, removed_words, len(images)
    except Exception as e:
        print(f"Error al procesar el archivo {pdf_path}: {e}")
        return [], [], 0

# Modificar la función principal

def build_inverted_index_parallel(folder_path, stats_path, posting_json_path, used_words_json_path, stopwords_json_path):
    """Construir el índice invertido con las nuevas funcionalidades."""
    inverted_index = {}
    doc_count = 0
    word_doc_count = {}
    unique_used_words = set()
    unique_removed_words = set()
    word_frequencies = Counter()
    stopword_frequencies = Counter()

    pdf_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.pdf')]

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pdf_file, pdf_path): pdf_path for pdf_path in pdf_files}
        for future in as_completed(futures):
            try:
                pdf_path = futures[future]
                filename = os.path.basename(pdf_path)
                used_words, removed_words, num_pages = future.result()
                doc_count += 1

                unique_used_words.update(set(used_words))
                unique_removed_words.update(set(removed_words))
                word_frequencies.update(used_words)
                stopword_frequencies.update(removed_words)

                # Calcular TF y actualizar índice invertido
                word_counts = Counter(used_words)
                total_words = len(used_words)

                for word, count in word_counts.items():
                    if word not in inverted_index:
                        inverted_index[word] = []
                    tf = count / total_words
                    inverted_index[word].append({"document": filename, "tf": tf})

                    word_doc_count[word] = word_doc_count.get(word, 0) + 1
            except Exception as e:
                print(f"Error: {e}")

    # Calcular IDF y TF-IDF
    for word, docs in inverted_index.items():
        idf = log(doc_count / word_doc_count[word])
        for entry in docs:
            entry["tf_idf"] = entry["tf"] * idf

    # Estadísticas para listas de postings
    posting_lengths = [(word, len(docs)) for word, docs in inverted_index.items()]
    posting_lengths.sort(key=lambda x: x[1], reverse=True)
    longest_posting = posting_lengths[0] if posting_lengths else (None, 0)
    shortest_postings = [word for word, length in posting_lengths if length == 1]
    average_posting = sum(length for _, length in posting_lengths) / len(posting_lengths) if posting_lengths else 0

    # Guardar JSON con información de postings
    posting_data = {
        "longest_posting": longest_posting,
        "shortest_postings": shortest_postings,
        "average_posting": average_posting,
        "postings": posting_lengths
    }
    with open(posting_json_path, 'w', encoding='utf-8') as f:
        json.dump(posting_data, f, ensure_ascii=False, indent=4)

    # Guardar palabras usadas en JSON
    used_words_data = word_frequencies.most_common()
    with open(used_words_json_path, 'w', encoding='utf-8') as f:
        json.dump(used_words_data, f, ensure_ascii=False, indent=4)

    # Guardar stopwords en JSON
    stopwords_data = stopword_frequencies.most_common()
    with open(stopwords_json_path, 'w', encoding='utf-8') as f:
        json.dump(stopwords_data, f, ensure_ascii=False, indent=4)

    # Guardar estadísticas generales
    stats = {
        "unique_used_words": len(unique_used_words),
        "unique_removed_words": len(unique_removed_words),
        "longest_posting": longest_posting,
        "shortest_postings": shortest_postings,
        "average_posting": average_posting
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    return inverted_index

# Llamada a la función principal
folder_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\decretos_2023"
stats_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\stats_test_full.json"
posting_json_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\posting_data_full.json"
used_words_json_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\used_words_full.json"
stopwords_json_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\stopwords_full.json"

inverted_index = build_inverted_index_parallel(folder_path, stats_path, posting_json_path, used_words_json_path, stopwords_json_path)
