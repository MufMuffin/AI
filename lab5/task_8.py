import os
import chromadb
from google import genai

API_KEY = "" 
client = genai.Client(api_key=API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_collection(name="python_docs")


def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values

def answer_without_rag(query: str) -> str:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=query,
    )
    return response.text.strip()

def answer_with_rag(query: str, max_distance: float = 0.62) -> tuple[str, list]:
    query_vector = get_embedding(query)
    results = collection.query(query_embeddings=[query_vector], n_results=3)

    retrieved_chunks = []
    if results and results['documents'] and results['documents'][0]:
        for idx, dist in enumerate(results['distances'][0]):
            if idx < 2 or dist <= max_distance:
                retrieved_chunks.append({
                    "id": results['ids'][0][idx],
                    "content": results['documents'][0][idx],
                    "source": results['metadatas'][0][idx]['source'],
                    "distance": dist
                })

    context_text = "\n\n".join([f"[{c['source']} | {c['id']}]: {c['content']}" for c in retrieved_chunks])

    prompt = f"""Використовуючи наведений нижче контекст, надай чітку та точну відповідь на запитання користувача.
Якщо відповіді на запитання немає в наданому контексті, обов'язково повідом про це.

Контекст:
{context_text}

Запитання:
{query}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )
    return response.text.strip(), retrieved_chunks


if __name__ == "__main__":
    test_query = "Як працюють текстові данні в Python"


    print(f"ТЕСТОВЕ ЗАПИТАННЯ: \"{test_query}\"\n")

    print("[1] Відповідь LLM БЕЗ RAG (чиста модель):")
    ans_no_rag = answer_without_rag(test_query)
    print(ans_no_rag)


    print(" Відповідь LLM З ВИКОРИСТАННЯМ RAG (наша локальна база):")
    ans_rag, chunks = answer_with_rag(test_query)
    print(ans_rag)
