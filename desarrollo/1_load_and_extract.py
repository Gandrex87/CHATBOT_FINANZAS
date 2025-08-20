# Hito 1: Carga, Extracción y Análisis Inicial del Documento
# ---------------------------------------------------------
# Objetivo: Cargar el PDF y extraer su texto de la forma más limpia posible.
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader

def load_financial_report(file_path: str):
    """
    Carga un documento PDF desde la ruta de archivo especificada.

    Args:
        file_path (str): La ruta al archivo PDF.

    Returns:
        list: Una lista de objetos Document, donde cada uno representa una página.
              Retorna una lista vacía si la ruta no es válida o hay un error.
    """
    if not os.path.exists(file_path):
        print(f"Error: El archivo no se encontró en la ruta: {file_path}")
        return []

    print(f"Cargando documento desde: {file_path}...")
    loader = PyMuPDFLoader(file_path)
    
    try:
        documents = loader.load()
        print(f"¡Éxito! Se han cargado {len(documents)} páginas.")
        return documents
    except Exception as e:
        print(f"Ocurrió un error al cargar el documento: {e}")
        return []

def inspect_document_content(documents: list):
    """
    Realiza una inspección básica del contenido extraído para su verificación.

    Args:
        documents (list): La lista de documentos (páginas) cargados.
    """
    if not documents:
        print("No hay documentos para inspeccionar.")
        return

    # --- Verificación 1: Inspeccionar una página de texto estándar ---
    # Elegimos una página que probablemente contenga párrafos de texto.
    page_to_inspect = 131 # Puedes cambiar este número
    if len(documents) > page_to_inspect:
        print("\n" + "="*50)
        print(f"INSPECCIÓN DE LA PÁGINA {page_to_inspect + 1}")
        print("="*50)
        
        page_content = documents[page_to_inspect].page_content
        metadata = documents[page_to_inspect].metadata

        print(f"\n--- Metadatos de la página ---")
        print(metadata)

        print(f"\n--- Contenido extraído (primeros 500 caracteres) ---")
        print(page_content[:500])
        print("...")

        # Búsqueda de posibles problemas
        if "  " in page_content or "\n " in page_content:
            print("\n[AVISO] Se detectaron posibles espacios extra o saltos de línea mal formateados.")
        if documents[page_to_inspect].metadata.get('page') != page_to_inspect:
             print("\n[AVISO] El número de página en los metadatos no coincide con el índice.")

    # --- Verificación 2: Inspeccionar una página que podría contener una tabla ---
    # Las tablas son un desafío común. Su texto puede salir desordenado.
    page_with_table = 237 # Cambia esto a una página donde sepas que hay una tabla
    if len(documents) > page_with_table:
        print("\n" + "="*50)
        print(f"INSPECCIÓN DE PÁGINA CON POSIBLE TABLA (Página {page_with_table + 1})")
        print("="*50)
        print("El texto de las tablas puede aparecer desestructurado. Esto es normal en la extracción inicial.")
        print("--- Contenido extraído (primeros 500 caracteres) ---")
        print(documents[page_with_table].page_content[:500])
        print("...")


if __name__ == "__main__":
    # Carga las variables de entorno (útil para futuras API keys)
    load_dotenv()

    # --- CONFIGURACIÓN ---
    # Coloca la ruta a tu informe financiero aquí.
    # pdf_file_path = os.getenv("PDF_PATH")
    pdf_file_path = "bd_pdf\PLAN_GENERAL_DE_CONTABILIDAD.pdf" # <-- 

    # 1. Cargar el documento
    loaded_documents = load_financial_report(pdf_file_path)

    # 2. Inspeccionar el contenido para una verificación inicial
    if loaded_documents:
        inspect_document_content(loaded_documents)

