import textwrap
from typing import Literal
from langchain_core.messages import HumanMessage
from graph.state import RagConversationalState
from graph.config import (ollama_embeddings, llm, qdrant_client, QDRANT_COLLECTION_NAME, bm25_index, bm25_corpus, reranker)


# --- 3. Definir los Nodos del Grafo ---

def retrieve_documents(state: RagConversationalState) -> RagConversationalState:
    print("\n---(Nodo: Recuperando Documentos)---")
    question = state["messages"][-1].content
    
    # ### CAMBIO 3: Aumentamos el límite para obtener más candidatos para el reranker
    CANDIDATE_LIMIT = 20

    query_vector = ollama_embeddings.embed_query(question)
    qdrant_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME, query_vector=query_vector, limit=CANDIDATE_LIMIT
    )
    
    tokenized_query = question.lower().split(" ")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_n_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:CANDIDATE_LIMIT]
    
    final_docs = {}
    for hit in qdrant_results:
        doc_text = hit.payload['generated_context'] + "\n\n" + hit.payload['original_chunk']
        final_docs[doc_text] = hit.score
    for idx in top_n_indices:
        doc_text = bm25_corpus[idx]
        if doc_text not in final_docs:
            final_docs[doc_text] = bm25_scores[idx]

    # No necesitamos ordenar aquí, ya que el reranker se encargará de ello
    unique_documents = list(final_docs.keys())
    
    print(f"Documentos recuperados para reranking: {len(unique_documents)}")
    return {"documents": unique_documents}

# ### CAMBIO 4: Creamos el nuevo nodo para el Reranking
def rerank_documents(state: RagConversationalState) -> RagConversationalState:
    """Nodo de Reclasificación: filtra y ordena los documentos por relevancia."""
    print("---(Nodo: Reclasificando Documentos)---")
    question = state["messages"][-1].content
    documents = state["documents"]
    
    if not documents:
        return {"documents": []}

    # Creamos pares de [pregunta, documento] para el modelo
    pairs = [(question, doc) for doc in documents]
    
    # Predecimos los scores de relevancia
    scores = reranker.predict(pairs)
    
    # Combinamos documentos y scores, y los ordenamos
    reranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    # Nos quedamos con los 5 mejores documentos
    TOP_K = 5
    final_documents = [doc for doc, score in reranked_docs[:TOP_K]]
    
    print(f"Documentos después de reranking: {len(final_documents)}")
    return {"documents": final_documents}

def generate_answer(state: RagConversationalState) -> RagConversationalState:
    print("---(Nodo: Generando Respuesta)---")
    # ### CAMBIO 5: Obtenemos la pregunta y el historial completo del estado
    question = state["messages"][-1].content
    documents = state["documents"]
    
    # Formateamos el historial para incluirlo en el prompt
    chat_history_str = ""
    # Ignoramos el último mensaje (la pregunta actual) para el historial
    for msg in state["messages"][:-1]:
        role = "Usuario" if isinstance(msg, HumanMessage) else "Asistente"
        chat_history_str += f"{role}: {msg.content}\n"

    context_str = "\n\n---\n\n".join(documents)
    
    # ### CAMBIO 6: El prompt ahora incluye el historial de la conversación
    prompt = textwrap.dedent(f"""
        Eres un asistente experto en análisis de informes financieros. Tu tarea es responder a la pregunta del usuario basándote en el siguiente contexto y el historial de la conversación.

        HISTORIAL DE LA CONVERSACIÓN:
        {chat_history_str}

        CONTEXTO EXTRAÍDO DEL DOCUMENTO:
        {context_str}

        PREGUNTA ACTUAL DEL USUARIO:
        {question}

        RESPUESTA (sé conciso y responde solo basándote en la información proporcionada):
    """)
    
    response = llm.invoke(prompt)
    # Devolvemos la respuesta como un nuevo mensaje para que se añada al historial
    return {"messages": [response]}

def handle_no_documents(state: RagConversationalState) -> RagConversationalState:
    print("---(Nodo: No se encontraron documentos)---")
    response_text = "Lo siento, no he podido encontrar información relevante en el documento para responder a tu pregunta."
    # También devolvemos un mensaje para que se guarde en el historial
    return {"messages": [response_text]}

# --- 4. Definir la Lógica Condicional (Sin cambios en la lógica interna) ---

def decide_to_generate(state: RagConversationalState) -> Literal["generate", "handle_no_documents"]:
    print("---(Borde Condicional: ¿Hay documentos?)---")
    if state["documents"]:
        print("Decisión: Sí, continuar a la generación.")
        return "generate"
    else:
        print("Decisión: No, usar el manejador de fallback.")
        return "handle_no_documents"
