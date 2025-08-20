# Hito 3: Creación y Contextualización de Chunks Hijo (Versión con Ollama)
# -------------------------------------------------------------------------
# Objetivo: Dividir los documentos padre en chunks más pequeños (hijos) y
#           enriquecer cada uno con un contexto generado por un LLM local vía Ollama.
# Librerías necesarias:
# pip install langchain-community tiktoken

import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import json

# --- Funciones de los Hitos 1 y 2 (las reutilizamos) ---

def load_financial_report(file_path: str) -> list:
    if not os.path.exists(file_path): return []
    loader = PyMuPDFLoader(file_path)
    return loader.load()

def clean_page_text(page_content: str) -> str:
    cleaned_text = re.sub(r'\n\s*\n', '\n', page_content)
    cleaned_text = re.sub(r'^\s*\d+\s*$', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'Plan General de Contabilidad\s+\d+', '', cleaned_text)
    cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned_text)
    lines = cleaned_text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text.strip()

def segment_into_parent_documents(documents: list) -> list:
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
    MIN_LENGTH_FOR_FIRST_CHUNK = 100
    first_chunk_content = split_parts[0].strip()
    if first_chunk_content and len(first_chunk_content) > MIN_LENGTH_FOR_FIRST_CHUNK:
        parent_documents.insert(0, first_chunk_content)
    return parent_documents

# --- NUEVA FUNCIÓN PARA GUARDAR EL RESULTADO ---
def save_chunks_to_json(chunks: list, filename: str):
    """
    Guarda la lista de chunks contextualizados en un archivo JSON.
    """
    print(f"\nGuardando {len(chunks)} chunks en el archivo '{filename}'...")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        print("¡Guardado completado con éxito!")
    except Exception as e:
        print(f"Error al guardar el archivo JSON: {e}")
# --- NUEVAS FUNCIONES PARA EL HITO 3 ---

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
    """
    Divide los documentos padre, contextualiza cada chunk hijo y los enriquece.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        is_separator_regex=False,
    )

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
                response = contextualize_chain.invoke({
                    "parent_document": parent_doc_text,
                    "child_chunk": chunk
                })
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
    
    print(f"\n¡Proceso completado! Se han creado {len(all_contextualized_chunks)} chunks contextualizados.")
    return all_contextualized_chunks

def inspect_contextualized_chunks(contextualized_chunks: list):
    """
    Muestra una muestra de los chunks contextualizados para verificación.
    """
    if not contextualized_chunks:
        print("No hay chunks para inspeccionar.")
        return

    print("\n" + "="*50)
    print("INSPECCIÓN DE CHUNKS CONTEXTUALIZADOS (OLLAMA)")
    print("="*50)

    import random
    samples_to_show = min(3, len(contextualized_chunks))
    sample_chunks = random.sample(contextualized_chunks, samples_to_show)

    for i, chunk_data in enumerate(sample_chunks):
        print(f"\n--- Muestra #{i+1} (del Documento Padre #{chunk_data['parent_doc_index'] + 1}) ---")
        print("\n[Contexto Generado por el LLM]:")
        print(chunk_data['generated_context'])
        print("\n[Chunk Original]:")
        print(chunk_data['original_chunk'])
        print("\n[CHUNK CONTEXTUALIZADO FINAL]:")
        print(chunk_data['contextualized_chunk'])
        print("-" * 50)


if __name__ == "__main__":
    pdf_file_path = "bd_pdf\pgc.pdf" 
    # 1. Cargar y segmentar (reutilizando la lógica anterior)
    all_docs = load_financial_report(pdf_file_path)
    content_docs = all_docs[5:]
    parent_documents = segment_into_parent_documents(content_docs)

    # 2. Inicializar el LLM de Ollama para contextualizar
    # --- CONFIGURACIÓN DE OLLAMA ---
    ollama_base_url = "http://10.1.0.176:11434" # del servidor....
    
    # Especifica el nombre del modelo que has descargado en Ollama.
    # Ejemplos: "llama3", "mistral", "gemma:2b"
    ollama_model = "llama3.1:latest"
    output_filename = "contextualized_chunks.json" # Nombre del archivo de salida
    
    print(f"Conectando a Ollama en {ollama_base_url} con el modelo {ollama_model}...")
    
    llm = ChatOllama(base_url=ollama_base_url, model=ollama_model, temperature=0)

     # 3. Crear y contextualizar los chunks
    final_chunks = create_and_contextualize_chunks(parent_documents, llm)

    # 4. Guardar el resultado en un archivo JSON
    if final_chunks:
        save_chunks_to_json(final_chunks, output_filename)
        # 5. Inspeccionar el resultado (opcional, ahora que está guardado)
        inspect_contextualized_chunks(final_chunks)

