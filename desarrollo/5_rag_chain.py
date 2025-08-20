# Hito 5: Construcción de la Cadena RAG con LangGraph (Versión Final - Corregida)
# ---------------------------------------------------------------------------------
# Objetivo: Orquestar el flujo de recuperación y generación para responder preguntas.
# Librerías necesarias:
# pip install langgraph langchain-ollama

import pickle
import json
from typing import List, TypedDict
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings, ChatOllama
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, END

# --- CONFIGURACIÓN (Debe coincidir con el script de indexación) ---
# Archivos de datos pre-procesados
BM25_INDEX_FILE = "bm25_index.pkl"
CONTEXTUALIZED_CHUNKS_FILE = "contextualized_chunks.json"

# Configuración de Qdrant
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "informe_financiero_rag"

# Configuración de Ollama
OLLAMA_BASE_URL = "http://10.1.0.176:11434"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_GENERATION_MODEL = "llama3.1:latest" # Modelo para generar las respuestas

# --- 1. Cargar los recursos necesarios ---

def load_bm25_index(filename: str):
    """Carga el índice BM25 desde un archivo."""
    print("Cargando índice BM25...")
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_all_chunks(filename: str) -> list:
    """Carga todos los chunks desde el archivo JSON para usarlos con BM25."""
    print(f"Cargando todos los chunks desde '{filename}' para el corpus BM25...")
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# Inicializamos los recursos al arrancar el script
bm25_index = load_bm25_index(BM25_INDEX_FILE)
all_chunks_data = load_all_chunks(CONTEXTUALIZED_CHUNKS_FILE)
# Creamos una lista solo con el texto para que BM25 pueda buscar
bm25_corpus = [chunk['contextualized_chunk'] for chunk in all_chunks_data]

qdrant_client = QdrantClient(url=QDRANT_URL)
ollama_embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBEDDING_MODEL)
llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_GENERATION_MODEL, temperature=0)

# --- 2. Definir el Estado del Grafo ---

class RagGraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str

# --- 3. Definir los Nodos del Grafo ---

def retrieve_documents(state: RagGraphState) -> RagGraphState:
    """
    Nodo de recuperación: obtiene documentos de Qdrant y BM25.
    """
    print("---(Nodo: Recuperando Documentos)---")
    question = state["question"]
    
    # --- Búsqueda Vectorial (Qdrant) ---
    query_vector = ollama_embeddings.embed_query(question)
    
    # --- CORRECCIÓN: Volver a usar el método .search() ---
    # Este método acepta un vector pre-calculado, que es lo que necesitamos.
    qdrant_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=5 # Obtener los 5 mejores resultados semánticos
    )
    
    # --- Búsqueda por Palabras Clave (BM25) ---
    tokenized_query = question.lower().split(" ")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    # Obtener los 5 mejores resultados por palabra clave
    top_n_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
    
    # --- Fusión y Deduplicación ---
    final_docs = {}
    # Añadir resultados de Qdrant
    for hit in qdrant_results:
        # Usamos el payload que guardamos durante la indexación
        doc_text = hit.payload['generated_context'] + "\n\n" + hit.payload['original_chunk']
        final_docs[doc_text] = hit.score

    # Añadir resultados de BM25, evitando duplicados
    for idx in top_n_indices:
        doc_text = bm25_corpus[idx]
        if doc_text not in final_docs:
            final_docs[doc_text] = bm25_scores[idx] # Podríamos normalizar el score si quisiéramos

    # Ordenar por score (simple, se podría usar RRF para una fusión más avanzada)
    sorted_docs = sorted(final_docs.items(), key=lambda item: item[1], reverse=True)
    
    # Nos quedamos con los 5 mejores documentos únicos
    top_documents = [doc[0] for doc in sorted_docs[:5]]
    
    print(f"Documentos recuperados: {len(top_documents)}")
    return {"documents": top_documents, "question": question}

def generate_answer(state: RagGraphState) -> RagGraphState:
    """
    Nodo de generación: crea una respuesta usando el LLM con el contexto recuperado.
    """
    print("---(Nodo: Generando Respuesta)---")
    question = state["question"]
    documents = state["documents"]
    
    # Formatear el contexto para el prompt
    context_str = "\n\n---\n\n".join(documents)
    
    prompt = f"""
    Eres un asistente experto en análisis de informes financieros. Tu tarea es responder a la pregunta del usuario basándote estricta y únicamente en el siguiente contexto extraído del documento. No utilices ningún conocimiento externo. Si la respuesta no se encuentra en el contexto, indica claramente que no tienes información al respecto.

    CONTEXTO:
    {context_str}

    PREGUNTA:
    {question}

    RESPUESTA:
    """
    
    response = llm.invoke(prompt)
    
    return {"generation": response.content}

# --- 4. Construir y Compilar el Grafo ---

workflow = StateGraph(RagGraphState)

# Añadir los nodos al grafo
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

# Definir las conexiones (el flujo de trabajo)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compilar el grafo para tener una aplicación ejecutable
app = workflow.compile()
print("\nGrafo RAG compilado y listo para usar.")

# --- 5. Ejecutar y Probar ---

if __name__ == "__main__":
    print("--- Chatbot Financiero RAG ---")
    print("Escribe tu pregunta o 'salir' para terminar.")
    
    while True:
        user_question = input("\nPregunta: ")
        if user_question.lower() == 'salir':
            break
        
        # Ejecutar el grafo con la pregunta del usuario
        final_state = app.invoke({"question": user_question})
        
        print("\nRespuesta Generada:")
        print(final_state["generation"])
