# Hito 3: Creación y Contextualización de Chunks Hijo (Nuevo PDF - Corregido)
# --------------------------------------------------------------------------
# Objetivo: Cargar, segmentar, dividir en chunks y contextualizar el nuevo PDF,
#           guardando el resultado en un archivo JSON.
# Librerías necesarias:
# pip install langchain-community langchain-ollama tiktoken

import os
import re
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

# --- Funciones de Carga y Segmentación (Hitos 1 y 2) ---

def load_financial_report(file_path: str) -> list:
    if not os.path.exists(file_path): return []
    loader = PyMuPDFLoader(file_path)
    return loader.load()

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
    content_docs = documents[start_page:]
    for doc in content_docs:
        doc.page_content = clean_page_text(doc.page_content)
    full_text = "\n".join([doc.page_content for doc in content_docs])
    
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
    
    matches = list(title_pattern.finditer(full_text))
    parent_documents = []
    if not matches: return []

    for i in range(len(matches)):
        start_pos = matches[i].start()
        end_pos = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        document_content = full_text[start_pos:end_pos].strip()
        if len(document_content) > 200:
            parent_documents.append(document_content)
            
    print(f"Segmentación completada. Se han creado {len(parent_documents)} documentos padre.")
    return parent_documents

# --- Lógica del Hito 3 ---

CONTEXTUALIZE_PROMPT_TEMPLATE = """
<document>
{parent_document}
</document>
Aquí está el chunk que queremos situar dentro del documento completo:
<chunk>
{child_chunk}
</chunk>
Por favor, proporciona un contexto breve y conciso en una sola frase para situar este chunk dentro del documento general. El objetivo es mejorar la recuperación en búsquedas. Responde únicamente con la frase de contexto y nada más.
"""

def create_and_contextualize_chunks(parent_docs: list, llm: ChatOllama):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    contextualize_prompt = ChatPromptTemplate.from_template(CONTEXTUALIZE_PROMPT_TEMPLATE)
    contextualize_chain = contextualize_prompt | llm
    all_contextualized_chunks = []

    print("\nIniciando creación y contextualización de chunks hijo con Ollama...")
    for i, parent_doc_text in enumerate(parent_docs):
        print(f"Procesando Documento Padre #{i+1}/{len(parent_docs)}...")
        child_chunks = text_splitter.split_text(parent_doc_text)
        for j, chunk in enumerate(child_chunks):
            print(f"  - Contextualizando chunk hijo #{j+1}/{len(child_chunks)}...")
            try:
                response = contextualize_chain.invoke({"parent_document": parent_doc_text, "child_chunk": chunk})
                generated_context = response.content.strip()
                contextualized_chunk = f"{generated_context}\n\n{chunk}"
                all_contextualized_chunks.append({
                    "parent_doc_index": i,
                    "original_chunk": chunk,
                    "generated_context": generated_context,
                    "contextualized_chunk": contextualized_chunk
                })
            except Exception as e:
                print(f"    ERROR al contextualizar chunk: {e}")
                continue
    
    print(f"\n¡Proceso de contextualización completado! Se han creado {len(all_contextualized_chunks)} chunks.")
    return all_contextualized_chunks

def save_chunks_to_json(chunks: list, filename: str):
    print(f"\nGuardando {len(chunks)} chunks en el archivo '{filename}'...")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        print("¡Guardado completado con éxito!")
    except Exception as e:
        print(f"Error al guardar el archivo JSON: {e}")

if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    # --- CORRECCIÓN: Añadir una 'r' antes de las comillas para crear un "raw string" ---
    pdf_file_path = r"bd_pdf\PLAN_GENERAL_DE_CONTABILIDAD.pdf" # <-- LÍNEA CORREGIDA
    CONTENT_START_PAGE = 8 # Página donde termina el sumario
    output_filename = "pgc_contextualized_chunks.json" # Nuevo nombre de archivo

    # Configuración de Ollama
    ollama_base_url = "http://10.1.0.176:11434"
    ollama_model = "gpt-oss:20b" #llama3.1:latest ,  

    # --- EJECUCIÓN DEL PROCESO COMPLETO ---
    
    # 1. Cargar y segmentar
    print("--- PASO 1: Cargando y Segmentando Documento ---")
    all_docs = load_financial_report(pdf_file_path)
    parent_documents = []
    if all_docs:
        parent_documents = segment_into_parent_documents(all_docs, CONTENT_START_PAGE)

    if parent_documents:
        # 2. Inicializar el LLM de Ollama
        print("\n--- PASO 2: Inicializando LLM de Ollama ---")
        print(f"Conectando a Ollama en {ollama_base_url} con el modelo {ollama_model}...")
        llm = ChatOllama(base_url=ollama_base_url, model=ollama_model, temperature=0)

        # 3. Crear y contextualizar los chunks
        print("\n--- PASO 3: Creando y Contextualizando Chunks ---")
        final_chunks = create_and_contextualize_chunks(parent_documents, llm)

        # 4. Guardar el resultado en un archivo JSON
        if final_chunks:
            save_chunks_to_json(final_chunks, output_filename)
            print(f"\nProceso finalizado. Los chunks están guardados en '{output_filename}'.")
    else:
        print("No se crearon documentos padre. El proceso se detiene.")
