# --- 5. Construir y Compilar el Grafo ---
from IPython.display import Image, display
from langgraph.graph import StateGraph, END
from graph.state import RagConversationalState
from graph.nodes import retrieve_documents, rerank_documents, generate_answer, handle_no_documents, decide_to_generate
from graph.memory import get_memory


def build_sequential_graph ():
    workflow = StateGraph(RagConversationalState)

    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("rerank", rerank_documents) # <-- Nuevo nodo
    workflow.add_node("handle_no_documents", handle_no_documents)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank") # retrieve ahora va a rerank
    workflow.add_conditional_edges(
        "rerank", # El borde condicional ahora empieza desde rerank
        decide_to_generate,
        {
            "generate": "generate",
            "handle_no_documents": "handle_no_documents",
        }
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("handle_no_documents", END)

    # ### CAMBIO 7: Añadimos el checkpointer al compilar el grafo
    graph = workflow.compile(checkpointer=get_memory())
    print("\nGrafo RAG Conversacional compilado y listo para usar.")
    #display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    return graph

