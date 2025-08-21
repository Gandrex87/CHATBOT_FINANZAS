import textwrap
from qdrant_client import QdrantClient, models
import json
from typing import Literal
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from graph.state import RagGraphState
from graph.config import (bm25_corpus, bm25_index, all_chunks_data, ollama_embeddings,qdrant_client, QDRANT_COLLECTION_NAME,   llm_router, llm_generator, reranker)

# --- 3. Definir los Nodos del Grafo ---
def route_question(state: RagGraphState) -> RagGraphState:
    print("---(Nodo: Clasificando Pregunta)---")
    # Lee la pregunta del último mensaje
    question = state["messages"][-1].content
    routing_prompt = f"""
    Tu tarea es clasificar la pregunta de un usuario sobre contabilidad para determinar qué base de conocimiento consultar.
    Las opciones son:
    - 'legacy': para preguntas sobre el Plan General Contable de 1990.
    - 'actual': para preguntas sobre el Plan General Contable vigente (post-2007).
    - 'both': para preguntas que comparan ambos planes o preguntan sobre su evolución.

    Pregunta del usuario: "{question}"

    Analiza la pregunta y responde únicamente con un objeto JSON con la clave "datasource" y uno de los tres valores: "legacy", "actual", o "both".
    """
    try:
        response = llm_router.invoke(routing_prompt)
        router_output = json.loads(response.content)
        datasource = router_output.get("datasource", "actual")
    except (json.JSONDecodeError, AttributeError):
        datasource = "actual"

    print(f"Decisión del Router: {datasource}")
    return {"datasource": datasource}

def retrieve_documents(state: RagGraphState) -> RagGraphState:
    print(f"---(Nodo: Recuperando Documentos de '{state['datasource']}')---")
    # Lee la pregunta del último mensaje
    question = state["messages"][-1].content
    datasource = state["datasource"]
    
    # COMENTARIO: Aumentamos el límite para obtener más candidatos para el reranker
    CANDIDATE_LIMIT = 20

    qdrant_filter = None
    if datasource != "both":
        source_map = {"legacy": "PGC_1990", "actual": "PGC_actual"}
        qdrant_filter = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source_map[datasource]))])

    query_vector = ollama_embeddings.embed_query(question)
    qdrant_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=CANDIDATE_LIMIT,
        query_filter=qdrant_filter
    )
    
    tokenized_query = question.lower().split(" ")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    all_bm25_candidates = [(bm25_scores[i], i) for i in range(len(bm25_scores))]

    if datasource != "both":
        filtered_candidates = [(score, idx) for score, idx in all_bm25_candidates if all_chunks_data[idx]['source'] == datasource]
    else:
        filtered_candidates = all_bm25_candidates

    filtered_candidates.sort(key=lambda x: x[0], reverse=True)
    top_bm25_indices = [idx for score, idx in filtered_candidates[:CANDIDATE_LIMIT]]
    
    final_docs = {}
    for hit in qdrant_results:
        doc_text = all_chunks_data[hit.id]['contextualized_chunk']
        final_docs[doc_text] = hit.score

    for idx in top_bm25_indices:
        doc_text = bm25_corpus[idx]
        if doc_text not in final_docs:
            final_docs[doc_text] = bm25_scores[idx]

    # COMENTARIO: Ya no filtramos a los 5 mejores, pasamos la lista completa de candidatos.
    unique_documents = list(final_docs.keys())
    print(f"Documentos recuperados para reranking: {len(unique_documents)}")
    return {"documents": unique_documents}

# COMENTARIO: Reintroducimos el nodo rerank_documents
def rerank_documents(state: RagGraphState) -> RagGraphState:
    """Nodo de Reclasificación: filtra y ordena los documentos por relevancia."""
    print("---(Nodo: Reclasificando Documentos)---")
    # Lee la pregunta del último mensaje
    question = state["messages"][-1].content
    documents = state["documents"]
    
    if not documents:
        return {"documents": []}

    pairs = [(question, doc) for doc in documents]
    scores = reranker.predict(pairs)
    reranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    TOP_K = 5
    final_documents = [doc for doc, score in reranked_docs[:TOP_K]]
    
    print(f"Documentos después de reranking: {len(final_documents)}")
    return {"documents": final_documents}

# Reemplaza tu función 'generate_answer' con esta
def generate_answer(state: RagGraphState) -> RagGraphState:
    print("---(Nodo: Generando Respuesta)---")
    question = state["messages"][-1].content
    documents = state["documents"]
    
    context_str = "\n\n---\n\n".join(documents)
    prompt = f"""
    Eres un asistente experto en contabilidad. Responde a la pregunta del usuario basándote estricta y únicamente en el siguiente contexto. Si la pregunta es una comparación, asegúrate de usar la información de ambas fuentes si se proporciona. Si la respuesta no está en el contexto, indícalo claramente.

    CONTEXTO:
    {context_str}

    PREGUNTA:
    {question}

    RESPUESTA:
    """
    response = llm_generator.invoke(prompt)
    
    # Antes: return {"generation": response.content}
    # Ahora:
    return {"messages": [AIMessage(content=response.content)]}


# COMENTARIO: Nuevo nodo para el caso de no encontrar documentos
def handle_no_documents(state: RagGraphState) -> RagGraphState:
    response_text = "Lo siento, no he podido encontrar información relevante para responder a tu pregunta."
    
    # Antes: return {"generation": response_text}
    # Ahora:
    return {"messages": [AIMessage(content=response_text)]}

# COMENTARIO: Nuevo borde condicional
def documents_exist(state: RagGraphState) -> Literal["continue", "handle_no_docs"]:
    return "continue" if state["documents"] else "handle_no_docs"

