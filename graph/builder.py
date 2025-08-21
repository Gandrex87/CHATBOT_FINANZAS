# --- 5. Construir y Compilar el Grafo ---
from IPython.display import Image, display
from langgraph.graph import StateGraph, END
from graph.state import RagGraphState
from graph.nodes import rerank_documents, retrieve_documents, route_question, generate_answer, handle_no_documents, documents_exist
from graph.memory import get_memory


def build_sequential_graph ():

    workflow = StateGraph(RagGraphState)
    workflow.add_node("router", route_question)
    workflow.add_node("retriever", retrieve_documents)
    # COMENTARIO: Añadimos el nodo rerank
    workflow.add_node("rerank", rerank_documents)
    workflow.add_node("generator", generate_answer)
    workflow.add_node("handle_no_docs", handle_no_documents)

    # COMENTARIO: Actualizamos el flujo para incluir el reranker
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "rerank") # El retriever ahora va al reranker
    workflow.add_conditional_edges(
        "rerank", # La condición ahora empieza desde el reranker
        documents_exist,
        {
            "continue": "generator",
            "handle_no_docs": "handle_no_docs"
        }
    )
    workflow.add_edge("generator", END)
    workflow.add_edge("handle_no_docs", END)

    graph = workflow.compile()
    #display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    return graph

