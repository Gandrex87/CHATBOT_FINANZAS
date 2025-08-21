# main.py

import uuid
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

# Se importa el constructor del grafo desde tu módulo
from graph.builder import build_sequential_graph
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Inicialización de la Aplicación y el Grafo ---

app = FastAPI(
    title="Chatbot RAG API",
    description="Una API para interactuar con un chatbot RAG conversacional usando LangGraph.",
    version="1.0.0",
)

# Se construye el grafo una sola vez al iniciar, llamando a tu función.
langgraph_app = build_sequential_graph()

# --- 2. Definición de los Modelos de Datos (Pydantic) ---

class ChatRequest(BaseModel):
    user_input: str = Field(..., description="El mensaje o pregunta del usuario.", example="¿cuales son las cuentas de Capital?")
    conversation_id: Optional[str] = Field(None, description="El ID de la conversación para mantener el contexto.", example="a1b2c3d4-e5f6-7890-1234-567890abcdef")

class ChatResponse(BaseModel):
    assistant_response: str = Field(..., description="La respuesta generada por el chatbot.", example="Los ingresos netos fueron de $1.2 millones.")
    conversation_id: str = Field(..., description="El ID de la conversación para continuar el chat.", example="a1b2c3d4-e5f6-7890-1234-567890abcdef")

# --- 3. Creación del Endpoint de la API ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Recibe un mensaje de un usuario y devuelve la respuesta del chatbot,
    gestionando el historial de la conversación.
    """
    # Se asigna un ID de conversación si no existe uno.
    conv_id = request.conversation_id or str(uuid.uuid4())
    
    # Se prepara la configuración para la memoria de LangGraph.
    config = {"configurable": {"thread_id": conv_id}}
    
    # Se prepara la entrada para el grafo en el formato que espera tu nodo `retrieve`.
    input_data = {"messages": [HumanMessage(content=request.user_input)]}
    
    # Se invoca el grafo de forma asíncrona.
    final_state = await langgraph_app.ainvoke(input_data, config)
    
    # Se extrae la respuesta del último mensaje, añadido por tu nodo `generate_answer` o `handle_no_documents`.
    # Nos aseguramos de que el último mensaje sea del asistente (AIMessage) o un string simple.
    last_message = final_state["messages"][-1]
    if isinstance(last_message, AIMessage):
        response_content = last_message.content
    else:
        # Esto manejaría el caso de `handle_no_documents` si devuelve un string.
        response_content = str(last_message)
    
    # Se devuelve la respuesta final.
    return ChatResponse(
        assistant_response=response_content,
        conversation_id=conv_id
    )


@app.get("/")
def read_root():
    return {"status": "Chatbot RAG API is running"}

#uvicorn main:app --reload