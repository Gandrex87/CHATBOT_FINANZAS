# # frontend.py

# import streamlit as st
# import requests
# import json
# import os

# # --- Configuración de la Página ---
# st.set_page_config(
#     page_title="Chatbot Finanzas RAG",
#     page_icon="🤖",
#     layout="centered"
# )

# # --- URL Configurable de la API ---
# # Lee la variable de entorno 'API_URL'. Si no existe, usa el valor por defecto.
# API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")


# # --- Título y Descripción ---
# st.title("Chatbot Finanzas RAG")
# st.caption("Este chatbot utiliza un modelo de lenguaje avanzado y RAG para responder preguntas sobre el Plan General de Contabilidad.")


# # ### NUEVO: Barra Lateral con el Botón de Reinicio ###
# with st.sidebar:
#     st.header("Opciones")
#     # Creamos un botón llamado "Reiniciar Conversación".
#     # `on_click` llama a la función `reset_conversation` cuando se presiona.
#     if st.button("Reiniciar Conversación", type="primary"):
#         # Limpiamos el estado de la sesión.
#         st.session_state.messages = [{"role": "assistant", "content": "Hola, ¿en qué puedo ayudarte hoy?"}]
#         st.session_state.conversation_id = None
#         # Mostramos una notificación para confirmar.
#         st.success("Conversación reiniciada.")


# # --- Gestión del Estado de la Sesión ---
# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "Hola, puedes hacerme preguntas sobre el Plan de Gestión Contable."}]

# if "conversation_id" not in st.session_state:
#     st.session_state.conversation_id = None

# # --- Mostrar el Historial del Chat ---
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # --- Entrada del Usuario ---
# if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    
#     st.chat_message("user").markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     payload = {
#         "user_input": prompt,
#         "conversation_id": st.session_state.conversation_id
#     }

#     with st.chat_message("assistant"):
#         with st.spinner("Pensando..."):
#             try:
#                 response = requests.post(API_URL, json=payload)
#                 response.raise_for_status()
                
#                 data = response.json()
#                 assistant_response = data["assistant_response"]
                
#                 st.session_state.conversation_id = data["conversation_id"]
                
#                 st.markdown(assistant_response)
#                 st.session_state.messages.append({"role": "assistant", "content": assistant_response})

#             except requests.exceptions.RequestException as e:
#                 error_message = f"Error al contactar la API: {e}"
#                 st.error(error_message)
#                 st.session_state.messages.append({"role": "assistant", "content": error_message})

# frontend.py (Versión final y refinada)

# frontend.py (Versión final con disclaimer)

import streamlit as st
import requests
import os

# --- Configuración ---
st.set_page_config(
    page_title="Chatbot Finanzas RAG",
    page_icon="🤖",
    layout="centered"
)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "Hola, puedes hacerme preguntas sobre el Plan General de Contabilidad."
}

# --- Funciones ---
def reset_conversation():
    """Reinicia el historial del chat y el ID de la conversación."""
    st.session_state.messages = [WELCOME_MESSAGE]
    st.session_state.conversation_id = None
    st.success("Conversación reiniciada.")

# --- Barra Lateral ---
with st.sidebar:
    st.header("Opciones")
    st.button("Reiniciar Conversación", on_click=reset_conversation, type="primary")

# --- Gestión del Estado de la Sesión ---
if "messages" not in st.session_state:
    st.session_state.messages = [WELCOME_MESSAGE]
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# --- Título y Descripción ---
st.title("🤖 Chatbot Finanzas RAG")
st.caption("Este chatbot utiliza un modelo de lenguaje avanzado y RAG para responder preguntas sobre el Plan General de Contabilidad.")

# --- MENSAJE DE ADVERTENCIA AÑADIDO AQUÍ ---
st.caption("El chatbot puede cometer errores, así que comprueba sus respuestas.")

# --- Mostrar el Historial del Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Entrada del Usuario ---
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    payload = {
        "user_input": prompt,
        "conversation_id": st.session_state.conversation_id
    }

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                
                data = response.json()
                assistant_response = data["assistant_response"]
                
                st.session_state.conversation_id = data["conversation_id"]
                
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            except requests.exceptions.RequestException as e:
                error_message = f"Error al contactar la API: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})