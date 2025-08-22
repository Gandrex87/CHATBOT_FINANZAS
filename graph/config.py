import pickle
import json
import sys
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings, ChatOllama
from rank_bm25 import BM25Okapi
import os # Necesario para las variables de entorno

# BM25_INDEX_FILE = "bm25_index_unificado.pkl"
# ALL_CHUNKS_UNIFIED_FILE = "all_chunks_unificado.json"
# QDRANT_URL = "http://localhost:6333"
# QDRANT_COLLECTION_NAME = "contabilidad_unificada"
# OLLAMA_BASE_URL = "http://10.1.0.176:11434"
# OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
# OLLAMA_GENERATION_MODEL = "gpt-oss:20b" #gpt-oss:20b
# OLLAMA_ROUTING_MODEL = "llama3.1:latest"

# --- Rutas de Ficheros (relativas al WORKDIR /app en Docker) ---
BM25_INDEX_FILE = "bm25_index_unificado.pkl"
ALL_CHUNKS_UNIFIED_FILE = "all_chunks_unificado.json"

# --- Configuración de Qdrant ---
# Leemos la URL del entorno. Usamos host.docker.internal como default para desarrollo local con Docker.
QDRANT_URL = os.getenv("QDRANT_URL", "http://host.docker.internal:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "contabilidad_unificada")

# --- Configuración de Ollama ---
# Leemos TODA la configuración de Ollama desde las variables de entorno
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "gpt-oss:20b")
OLLAMA_ROUTING_MODEL = os.getenv("OLLAMA_ROUTING_MODEL", "llama3.1:latest")

# --- Instancias de Clientes y Modelos ---
# Ahora tu código que usa estas variables será portable y configurable.
# Por ejemplo:
# client = QdrantClient(url=QDRANT_URL)
# llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_GENERATION_MODEL)

# --- 1. Cargar los recursos necesarios ---
def load_resources():
    print("Cargando recursos...")
    bm25_index = pickle.load(open(BM25_INDEX_FILE, "rb"))
    all_chunks_data = json.load(open(ALL_CHUNKS_UNIFIED_FILE, 'r', encoding='utf-8'))
    bm25_corpus = [chunk['contextualized_chunk'] for chunk in all_chunks_data]
    qdrant_client = QdrantClient(url=QDRANT_URL)
    ollama_embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBEDDING_MODEL)
    llm_generator = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_GENERATION_MODEL, temperature=0)
    llm_router = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_ROUTING_MODEL, temperature=0, format="json")
    
    # COMENTARIO: Volvemos a cargar el modelo reranker
    reranker = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2', max_length=512)
    
    print("Recursos cargados.")
    return bm25_index, all_chunks_data, bm25_corpus, qdrant_client, ollama_embeddings, llm_generator, llm_router, reranker

bm25_index, all_chunks_data, bm25_corpus, qdrant_client, ollama_embeddings, llm_generator, llm_router, reranker = load_resources()

