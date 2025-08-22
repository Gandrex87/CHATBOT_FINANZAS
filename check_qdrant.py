# check_qdrant.py (Versión corregida)
from qdrant_client import QdrantClient

# Asegúrate de que la URL es la correcta para tu entorno Docker Compose
QDRANT_URL = "http://localhost:6333" 
COLLECTION_NAME = "contabilidad_unificada"

try:
    client = QdrantClient(url=QDRANT_URL)
    print(f"Conectado a Qdrant en {QDRANT_URL}")

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)

    # --- INICIO DE LA CORRECCIÓN ---
    # Guardamos el conteo en una variable para manejar el caso de que sea None
    count = collection_info.vectors_count
    
    # Imprimimos el valor para saber qué estamos recibiendo exactamente
    print(f"\nInformación de la colección '{COLLECTION_NAME}':")
    print(f"- Conteo de vectores: {count}")

    # Comprobamos primero si el conteo no es None antes de compararlo con 0
    if count is not None and count > 0:
    # --- FIN DE LA CORRECCIÓN ---
        print("\n--- Mostrando 2 puntos de ejemplo ---")
        records = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=2,
            with_payload=True,
            with_vectors=False
        )
        for i, record in enumerate(records[0]):
            print(f"\n--- Punto de ejemplo {i+1} ---")
            print(f"ID: {record.id}")
            print("Payload (Datos guardados):")
            print(record.payload)
    else:
        print("\nLa colección está vacía o no tiene vectores.")

except Exception as e:
    print(f"\nError: {e}")