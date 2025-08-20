# Hito 2: Limpieza de Texto y Segmentación Lógica (Nuevo PDF - v10 Definitivo)
# --------------------------------------------------------------------------
# Objetivo: Utilizar un método de búsqueda por posición (finditer) para
#           lograr una segmentación precisa y a prueba de errores.

import os
import re
from langchain_community.document_loaders import PyMuPDFLoader

# --- Función del Hito 1 (la reutilizamos) ---
def load_financial_report(file_path: str) -> list:
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

# --- Función de limpieza (sin cambios) ---
def clean_page_text(page_content: str) -> str:
    cleaned_text = re.sub(r'\n\s*\n', '\n', page_content)
    cleaned_text = re.sub(r'–\s*\d+\s*–', '', cleaned_text)
    cleaned_text = re.sub(r'^\s*\d+\s*$', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned_text)
    lines = cleaned_text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text.strip()


def segment_into_parent_documents(documents: list, start_page: int) -> list:
    """
    Segmenta los documentos encontrando las posiciones de los títulos principales.
    """
    print(f"\nIniciando limpieza y segmentación definitiva a partir de la página {start_page}...")
    
    content_docs = documents[start_page:]
    
    for doc in content_docs:
        doc.page_content = clean_page_text(doc.page_content)

    full_text = "\n".join([doc.page_content for doc in content_docs])

    # --- PATRÓN DE TÍTULOS DEFINITIVO ---
    # Este patrón busca el inicio de cada sección principal que queremos.
    title_pattern = re.compile(
        r'^(?:PRESENTACIÓN|'
        r'REAL DECRETO 1514/07|'
        r'Real Decreto 1159/2010|'
        r'Real Decreto 602/2016|'
        r'Real Decreto 1/2021|'
        r'PRIMERA PARTE|'
        r'SEGUNDA PARTE|'
        r'TERCERA PARTE|'
        r'CUARTA PARTE|'
        r'QUINTA PARTE)',
        flags=re.MULTILINE
    )

    # Encontrar todos los inicios de títulos
    matches = list(title_pattern.finditer(full_text))
    
    parent_documents = []
    
    if not matches:
        print("No se encontraron títulos para segmentar.")
        return []

    # Crear un documento por cada sección encontrada
    for i in range(len(matches)):
        start_pos = matches[i].start()
        # El final de la sección es el inicio de la siguiente, o el final del texto
        end_pos = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        
        document_content = full_text[start_pos:end_pos].strip()
        
        # Filtro final para asegurar calidad
        if len(document_content) > 200:
            parent_documents.append(document_content)
            
    print(f"Segmentación completada. Se han creado {len(parent_documents)} documentos padre.")
    return parent_documents


def inspect_parent_documents(parent_docs: list):
    """
    Muestra información sobre los documentos padre creados para verificación.
    """
    if not parent_docs:
        print("No se crearon documentos padre.")
        return

    print("\n" + "="*50)
    print("INSPECCIÓN DE DOCUMENTOS PADRE (DEFINITIVO)")
    print("="*50)
    
    # Mostramos todos los documentos padre para una verificación completa
    for i, doc_text in enumerate(parent_docs):
        print(f"\n--- Documento Padre #{i+1} ---")
        print(f"Título/Inicio: {doc_text[:120].replace(chr(10), ' ')}...")
        print(f"Longitud (caracteres): {len(doc_text)}")


if __name__ == "__main__":
    pdf_file_path = "bd_pdf\PLAN_GENERAL_DE_CONTABILIDAD.pdf" # <-- CAMBIA ESTA LÍNEA

    # Página donde termina el SUMARIO y empieza el contenido (ej. "PRESENTACIÓN")
    # Si es la página 9, el índice a usar es 8.
    CONTENT_START_PAGE = 8 

    # 1. Cargar el documento
    loaded_docs = load_financial_report(pdf_file_path)

    if loaded_docs:
        # 2. Limpiar y segmentar en documentos padre
        parent_documents_list = segment_into_parent_documents(loaded_docs, CONTENT_START_PAGE)
        
        # 3. Inspeccionar el resultado
        inspect_parent_documents(parent_documents_list)
