# --- 2. Definir el Estado del Grafo ---
from typing import List, TypedDict, Literal, Sequence, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ### Actualizamos el estado para incluir el historial de mensajes: # La clave 'messages' es especial. `add_messages` se asegura de que los nuevos mensajes se añadan a la lista en lugar de sobreescribirla.
class RagConversationalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: List[str]