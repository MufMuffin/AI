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

def retrieve_dynamic_chunks(query: str, max_distance: float = 0.55, max_k: int = 5):
    query_vector = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=max_k
    )

    retrieved_chunks = []
    
    if results and results['documents'] and results['documents'][0]:
        documents = results['documents'][0]
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results else [0] * len(ids)

        for idx, dist in enumerate(distances):
            if dist <= max_distance:
                retrieved_chunks.append({
                    "id": ids[idx],
                    "content": documents[idx],
                    "source": metadatas[idx]['source'],
                    "distance": dist
                })

        if not retrieved_chunks:
            retrieved_chunks.append({
                "id": ids[0],
                "content": documents[0],
                "source": metadatas[0]['source'],
                "distance": distances[0]
            })

    return retrieved_chunks

def answer_with_rag(query: str, max_distance: float = 0.55):
    chunks = retrieve_dynamic_chunks(query, max_distance=max_distance)

    context_text = "\n\n".join([f"[{c['source']} | {c['id']} | Distance: {c['distance']:.4f}]:\n{c['content']}" for c in chunks])

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

    return response.text, chunks

if __name__ == "__main__":
    test_questions = [
        "Що таке Python і хто його створив?",
        "Що таке списки та кортежі та яка їх відмінність",
        "Що таке концепція Качиної типізації (Duck Typing) в ООП?",
        "Для чого використовуються фреймворки LangChain та LlamaIndex?",
        "Що таке векторні сховища і для чого призначені Chroma та FAISS?",
        "Як працюють модифікатори доступу public, private та protected у мові C++?"  #додаькове питання для перевірки обмежень RAG
    ]

    for idx, q in enumerate(test_questions, 1):
        print(f"ЗАПИТ #{idx}: {q}")
        answer, retrieved_chunks = answer_with_rag(q, max_distance=0.55)

        print(f"\n[Знайдено релевантних чанків: {len(retrieved_chunks)}]")
        for c in retrieved_chunks:
            print(f"  • ID: {c['id']} | Джерело: {c['source']} ")

        print("\n[Відповідь LLM (Gemini)]:")
        print(answer)