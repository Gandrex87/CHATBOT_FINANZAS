import pickle
import json
import sys
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings, ChatOllama
from rank_bm25 import BM25Okapi
import os # Necesario para las variables de entorno

# --- CONFIGURACIÓN ESTATICA ---
BM25_INDEX_FILE = "bm25_index.pkl"
CONTEXTUALIZED_CHUNKS_FILE = "contextualized_chunks.json"
QDRANT_COLLECTION_NAME = "informe_financiero_rag"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_GENERATION_MODEL = "gpt-oss:20b" #  gpt-oss:20b  llama3.1:latest

# --- CONFIGURACIÓN DINÁMICA (CON VARIABLES DE ENTORNO) ---
# Lee la variable de entorno 'QDRANT_URL'. Si no existe, usa "http://localhost:6333".
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Lee la variable de entorno 'OLLAMA_BASE_URL'. Si no existe, usa tu IP local.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://10.1.0.176:11434")

# --- 1. Cargar los recursos necesarios ---
# Esta función ahora usará las variables QDRANT_URL y OLLAMA_BASE_URL que acabamos de definir,
# por lo que no necesita ningún cambio interno.
def load_resources():
    try:
        with open(BM25_INDEX_FILE, "rb") as f:
            bm25_index = pickle.load(f)
        with open(CONTEXTUALIZED_CHUNKS_FILE, 'r', encoding='utf-8') as f:
            all_chunks_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo necesario: {e.filename}", file=sys.stderr)
        sys.exit(1)

    bm25_corpus = [chunk['contextualized_chunk'] for chunk in all_chunks_data]
    qdrant_client = QdrantClient(url=QDRANT_URL)
    ollama_embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBEDDING_MODEL)
    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_GENERATION_MODEL, temperature=0)

    print("Cargando modelo Reranker...")
    reranker = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2', max_length=512)
    print("Recursos cargados correctamente.")

    return bm25_index, bm25_corpus, qdrant_client, ollama_embeddings, llm, reranker

bm25_index, bm25_corpus, qdrant_client, ollama_embeddings, llm, reranker = load_resources()