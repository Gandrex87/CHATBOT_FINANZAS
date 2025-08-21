# --- 2. Definir el Estado del Grafo ---
from typing import List, TypedDict, Literal, Sequence, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ### Actualizamos el estado para incluir el historial de mensajes: # La clave 'messages' es especial. `add_messages` se asegura de que los nuevos mensajes se añadan a la lista en lugar de sobreescribirla.
class RagGraphState(TypedDict):
    """
    El estado del grafo conversacional.

    Atributos:
        messages: La lista de mensajes que forman la conversación. La anotación
                  hace que los nuevos mensajes se añadan en lugar de reemplazar.
        documents: La lista de documentos recuperados para usar como contexto.
        datasource: La fuente de datos decidida por el router.
    """
    # La clave 'messages' gestionará todo el historial de la conversación.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Las otras claves se mantienen para los pasos intermedios.
    documents: List[str]
    datasource: Literal["legacy", "actual", "both"]