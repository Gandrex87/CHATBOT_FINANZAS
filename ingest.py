# ingest.py

import json
import os
import pickle
import sys
import time
from qdrant_client import QdrantClient, models
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

# --- CONFIGURACIÓN ---
# Mejor práctica: hacemos los nombres de los archivos también configurables
INPUT_JSON_FILE = os.getenv("INPUT_JSON_FILE", "all_chunks_unificado.json")
BM25_INDEX_FILE = os.getenv("BM25_INDEX_FILE", "bm25_index_unificado.pkl")
QDRANT_COLLECTION_NAME = "contabilidad_unificada"
VECTOR_DIMENSION = 768

# Leemos la configuración desde variables de entorno con valores por defecto
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://10.1.0.176:11434")
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest" 




def check_if_data_exists(client: QdrantClient) -> bool:
    """Comprueba si la colección ya existe y tiene datos."""
    try:
        # Intenta obtener información de la colección
        collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        # Comprueba si hay vectores en la colección
        if collection_info.vectors_count > 0:
            print(f"La colección '{QDRANT_COLLECTION_NAME}' ya existe y contiene {collection_info.vectors_count} vectores.")
            return True
        else:
            print(f"La colección '{QDRANT_COLLECTION_NAME}' existe pero está vacía.")
            return False
    except Exception:
        # La excepción se lanza si la colección no existe
        print(f"La colección '{QDRANT_COLLECTION_NAME}' no existe.")
        return False

def wait_for_qdrant(client: QdrantClient, timeout: int = 60):
    """Espera a que Qdrant esté disponible antes de continuar."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Intenta hacer una petición simple a Qdrant
            client.get_collections()
            print("Qdrant está listo y disponible.")
            return True
        except Exception:
            print("Esperando a que Qdrant esté disponible...")
            time.sleep(2)
    print("Error: No se pudo conectar a Qdrant después de 60 segundos.")
    return False

def load_chunks_from_json(filename: str) -> list:
    """Carga los chunks desde el archivo JSON."""
    print(f"Cargando chunks desde '{filename}'...")
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_bm25_index(chunks: list):
    """Crea y guarda un índice BM25."""
    print("\nCreando índice BM25...")
    corpus = [chunk['contextualized_chunk'] for chunk in chunks]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Índice BM25 guardado en '{BM25_INDEX_FILE}'.")

def index_in_qdrant(client: QdrantClient, chunks: list, embeddings_model: OllamaEmbeddings):
    """Genera embeddings y los indexa en Qdrant en lotes."""
    print("\nIniciando indexación en Qdrant...")
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
    )
    print(f"Colección '{QDRANT_COLLECTION_NAME}' creada/recreada.")

    batch_size = 64
    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i+batch_size]
        texts_to_embed = [chunk['contextualized_chunk'] for chunk in batch_chunks]
        
        print(f"Procesando lote {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...")
        embeddings = embeddings_model.embed_documents(texts_to_embed)
        
        points_to_upload = [
            models.PointStruct(
                id=i+j,
                vector=embeddings[j],
                payload={
                    "original_chunk": chunk_data["original_chunk"],
                    "generated_context": chunk_data["generated_context"],
                    "parent_doc_index": chunk_data["parent_doc_index"]
                }
            ) for j, chunk_data in enumerate(batch_chunks)
        ]
        
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points_to_upload,
            wait=True
        )
    print("\n¡Indexación en Qdrant completada con éxito!")

if __name__ == "__main__":
    print("--- Proceso de Ingesta de Datos ---")
    
    qdrant_client = QdrantClient(url=QDRANT_URL)
    
    # Espera a que Qdrant esté listo antes de continuar
    if not wait_for_qdrant(qdrant_client):
        sys.exit(1) # Termina si Qdrant no está disponible

    # Comprueba si los datos ya existen
    if check_if_data_exists(qdrant_client):
        print("Saltando el proceso de ingesta.")
        sys.exit(0) # Termina el script con éxito
    
    print("\nLa base de datos está vacía. Iniciando el proceso de ingesta completo.")
    
    all_chunks = load_chunks_from_json(INPUT_JSON_FILE)
    if all_chunks:
        create_bm25_index(all_chunks)
        
        print(f"\nInicializando modelo de embeddings '{OLLAMA_EMBEDDING_MODEL}'...")
        ollama_embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBEDDING_MODEL
        )
        
        index_in_qdrant(qdrant_client, all_chunks, ollama_embeddings)
    else:
        print("No se encontraron chunks para procesar.")