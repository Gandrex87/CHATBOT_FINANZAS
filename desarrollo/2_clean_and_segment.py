# Hito 2: Limpieza de Texto y Segmentación Lógica
# ------------------------------------------------
# Objetivo: Limpiar el texto extraído y agruparlo en "documentos padre" lógicos.
# Librerías necesarias:
# pip install langchain-community pymupdf python-dotenv

import os
import re
from dotenv import load_dotenv
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

# --- NUEVAS FUNCIONES PARA EL HITO 2 ---

def clean_page_text(page_content: str) -> str:
    """
    Realiza una limpieza básica del texto de una página.

    Args:
        page_content (str): El texto extraído de una página.

    Returns:
        str: El texto limpiado.
    """
    # 1. Eliminar saltos de línea excesivos
    cleaned_text = re.sub(r'\n\s*\n', '\n', page_content)
    
    # 2. Eliminar líneas que son solo números (típicos pies de página)
    cleaned_text = re.sub(r'^\s*\d+\s*$', '', cleaned_text, flags=re.MULTILINE)
    
    # # 3. Eliminar encabezados o pies de página específicos (EJEMPLO)
    # cleaned_text = cleaned_text.replace("Informe Anual ACME Corp", "")

    # --- NUEVA REGLA DE LIMPIEZA AÑADIDA ---
    # 4. Eliminar el patrón "Plan General de Contabilidad" seguido de espacios y un número
    # Esto busca la frase literal, seguida de uno o más espacios (\s+), y uno o más dígitos (\d+)
    cleaned_text = re.sub(r'Plan General de Contabilidad\s+\d+', '', cleaned_text)
    # --- FIN DE LA NUEVA REGLA ---
    
    # 5. Unir palabras separadas por un guion al final de una línea
    cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned_text)
    
    # 6. Eliminar espacios en blanco al principio y final de cada línea
    lines = cleaned_text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    cleaned_text = '\n'.join(cleaned_lines)

    return cleaned_text.strip()


def segment_into_parent_documents(documents: list) -> list:
    """
    Segmenta los documentos cargados en "documentos padre" basados en la estructura
    del índice del documento (ej. 1.1, 1.2, etc.).
    """
    print("\nIniciando limpieza y segmentación...")
    
    for doc in documents:
        doc.page_content = clean_page_text(doc.page_content)

    full_text = "\n".join([doc.page_content for doc in documents])

    split_pattern = r'(^\d+\.\d+\s)'
    
    split_parts = re.split(split_pattern, full_text, flags=re.MULTILINE)
    
    parent_documents = []
    i = 1
    while i < len(split_parts):
        document_content = split_parts[i] + split_parts[i+1]
        parent_documents.append(document_content.strip())
        i += 2
        
    # El primer trozo (antes de la sección 1.1) puede ser ruido.
    # Lo añadimos solo si tiene una longitud mínima para ser considerado contenido real.
    MIN_LENGTH_FOR_FIRST_CHUNK = 100 # Umbral de caracteres
    first_chunk_content = split_parts[0].strip()
    
    if first_chunk_content and len(first_chunk_content) > MIN_LENGTH_FOR_FIRST_CHUNK:
        print(f"Añadiendo el fragmento inicial (longitud: {len(first_chunk_content)}) como documento padre #1.")
        parent_documents.insert(0, first_chunk_content)
    else:
        print(f"Ignorando el fragmento inicial por ser demasiado corto (longitud: {len(first_chunk_content)}).")


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
    print("INSPECCIÓN DE DOCUMENTOS PADRE (Nivel 2)")
    print("="*50)
    
    for i, doc_text in enumerate(parent_docs[:25]):
        print(f"\n--- Documento Padre #{i+1} ---")
        print(f"Título/Inicio: {doc_text[:100].replace(chr(10), ' ')}...")
        print(f"Longitud (caracteres): {len(doc_text)}")
    
    if len(parent_docs) > 5:
        print("\n...")
        print(f"(Total: {len(parent_docs)} documentos padre)")


if __name__ == "__main__":
    load_dotenv()
    pdf_file_path = "bd_pdf\pgc.pdf" # <-- CAMBIA ESTA LÍNEA

    # 1. Cargar el documento
    loaded_docs = load_financial_report(pdf_file_path)

    if loaded_docs:
        # --- NUEVO PASO: IGNORAR LAS PÁGINAS DEL ÍNDICE ---
        # El índice termina en la página 5, lo que corresponde a los índices 0, 1, 2, 3, 4.
        # Nos quedamos con los documentos a partir del índice 5 (la sexta página).
        print(f"Ignorando las primeras 5 páginas (índice)...")
        content_docs = loaded_docs[5:]
        # --- FIN DEL NUEVO PASO ---

        # 2. Limpiar y segmentar en documentos padre (usando solo el contenido)
        parent_documents_list = segment_into_parent_documents(content_docs)
        
        # 3. Inspeccionar el resultado
        inspect_parent_documents(parent_documents_list)
