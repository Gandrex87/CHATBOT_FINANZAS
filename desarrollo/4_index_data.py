# Hito 4: Indexación Híbrida (Qdrant + BM25) - Versión Corregida
# -----------------------------------------------------------------
# Objetivo: Cargar los chunks contextualizados y crear dos índices:
#           1. Un índice vectorial en Qdrant para búsqueda semántica.
#           2. Un índice BM25 para búsqueda por palabras clave.
# Librerías necesarias:
# pip install qdrant-client langchain-ollama rank_bm25 numpy

import json
import os
import pickle
from qdrant_client import QdrantClient, models
# --- CAMBIO 1: Importar desde el nuevo paquete ---
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np

# --- CONFIGURACIÓN ---
# Archivo de entrada con los chunks procesados
INPUT_JSON_FILE = "contextualized_chunks.json"

# Configuración de Qdrant
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "informe_financiero_rag"

# Configuración de Ollama para Embeddings
OLLAMA_BASE_URL = "http://10.1.0.176:11434"
# Modelo recomendado para embeddings en español. Asegúrate de tenerlo: `ollama pull nomic-embed-text`
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest" 
# La dimensión del vector para nomic-embed-text es 768
VECTOR_DIMENSION = 768

# Archivo de salida para el índice BM25
BM25_INDEX_FILE = "bm25_index.pkl"

def load_chunks_from_json(filename: str) -> list:
    """Carga los chunks desde el archivo JSON de salida del hito anterior."""
    print(f"Cargando chunks desde '{filename}'...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Se han cargado {len(chunks)} chunks.")
        return chunks
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filename}'. Asegúrate de que existe.")
        return []
    except Exception as e:
        print(f"Error al cargar el archivo JSON: {e}")
        return []

def create_bm25_index(chunks: list):
    """Crea y guarda un índice BM25 a partir de los chunks."""
    print("\nIniciando creación del índice BM25...")
    
    # Extraemos el texto de los chunks contextualizados para el corpus
    corpus = [chunk['contextualized_chunk'] for chunk in chunks]
    
    # El tokenizador simple divide por espacios en blanco. Es suficiente para empezar.
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Guardamos el objeto bm25 en un archivo para su uso futuro
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25, f)
        
    print(f"Índice BM25 creado y guardado en '{BM25_INDEX_FILE}'.")

def index_in_qdrant(chunks: list, embeddings_model: OllamaEmbeddings):
    """Genera embeddings y los indexa en Qdrant en lotes."""
    print("\nIniciando indexación en Qdrant...")
    
    # 1. Inicializar el cliente de Qdrant
    client = QdrantClient(url=QDRANT_URL)
    
    # 2. Crear la colección si no existe
    try:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
        )
        print(f"Colección '{QDRANT_COLLECTION_NAME}' creada/recreada en Qdrant.")
    except Exception as e:
        print(f"Error al crear la colección en Qdrant: {e}")
        return

    # 3. Preparar los datos y subirlos en lotes
    batch_size = 64 # Puedes ajustar este tamaño según la memoria de tu máquina
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        # Extraemos el texto a "embeddear"
        texts_to_embed = [chunk['contextualized_chunk'] for chunk in batch_chunks]
        
        print(f"Procesando lote {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}... (chunks {i+1}-{min(i+batch_size, total_chunks)})")
        
        # Generar embeddings para el lote
        embeddings = embeddings_model.embed_documents(texts_to_embed)
        
        # Preparar los puntos para Qdrant
        points_to_upload = []
        for j, chunk_data in enumerate(batch_chunks):
            points_to_upload.append(
                models.PointStruct(
                    id=i+j, # ID único para cada punto
                    vector=embeddings[j],
                    payload={
                        # Guardamos el resto de la información como metadatos
                        "original_chunk": chunk_data["original_chunk"],
                        "generated_context": chunk_data["generated_context"],
                        "parent_doc_index": chunk_data["parent_doc_index"]
                    }
                )
            )
        
        # Subir el lote a Qdrant
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points_to_upload,
            wait=True # Esperar a que la operación se complete
        )
        
    print("\n¡Indexación en Qdrant completada con éxito!")


if __name__ == "__main__":
    # 1. Cargar los chunks procesados
    all_chunks = load_chunks_from_json(INPUT_JSON_FILE)
    
    if all_chunks:
        # 2. Crear y guardar el índice BM25
        create_bm25_index(all_chunks)
        
        # 3. Inicializar el modelo de embeddings de Ollama
        print(f"\nInicializando modelo de embeddings '{OLLAMA_EMBEDDING_MODEL}' desde Ollama...")
        ollama_embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBEDDING_MODEL
        )
        
        # 4. Indexar los datos en Qdrant
        index_in_qdrant(all_chunks, ollama_embeddings)
