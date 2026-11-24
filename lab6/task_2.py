import os
import sys
import chromadb
from google import genai

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks_output")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db_lab6")

TEST_DATA = [
    {"text": "Python є мовою програмування високого рівня з динамічною типізацією.", "label": "програмування"},
    {"text": "Нейронна мережа складається з набору взаємопов'язаних нейронів і навчається через зворотне поширення помилки.", "label": "штучний інтелект"},
    {"text": "Реляційна система управління даними зберігає інформацію у вигляді таблиць зі строгою схемою.", "label": "бази даних"},
    {"text": "Протокол TCP/IP забезпечує надійну доставку пакетів між вузлами в мережі.", "label": "комп'ютерні мережі"},
    {"text": "Модель Gemini Flash використовує архітектуру Transformer для обробки великого контексту.", "label": "штучний інтелект"},
    {"text": "Векторне сховище ChromaDB прискорює пошук найближчих векторів за допомогою індексів.", "label": "бази даних"},
    {"text": "Асинхронний фреймворк FastAPI дозволяє швидко створювати REST API на мові Python.", "label": "програмування"},
    {"text": "Маршрутизатор визначає оптимальний шлях для передачі мережевих пакетів за допомоги таблиці маршрутизації.", "label": "комп'ютерні мережі"},
    {"text": "Транзакції в базах даних повинні задовольняти вимогам ACID для забезпечення цілісності даних.", "label": "бази даних"},
    {"text": "Згорткові нейронні мережі CNN ефективно використовуються для задач комп'ютерного бачення.", "label": "штучний інтелект"}
]

def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return response.embeddings[0].values

def create_chunks_files():
    """Крок 1: Розбиття 4 вихідних документів на чанки та збереження в папку chunks_output."""
    file_names = ['doc1.txt', 'doc2.txt', 'doc3.txt', 'doc4.txt']
    chunks = []
    chunk_id = 1

    for file_name in file_names:
        file_path = os.path.join(BASE_DIR, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                raw_chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
                for paragraph in raw_chunks:
                    chunks.append({
                        "id": f"chunk_{chunk_id:02d}",
                        "content": paragraph,
                        "source": file_name
                    })
                    chunk_id += 1

    output_dir = CHUNKS_DIR
    os.makedirs(output_dir, exist_ok=True)

    for item in chunks:
        file_name = f"{item['id']}.txt"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {item['source']}\n")
            f.write(f"ID: {item['id']}\n")
            f.write("-" * 40 + "\n\n")
            f.write(item['content'])

    print(f"[+] Створено та збережено {len(chunks)} файлів у папку {output_dir}")

def load_chunks_to_chroma():
    """Крок 2: Зчитування файлів із chunks_output та запис у ChromaDB."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_or_create_collection(name="lab6_docs")

    if collection.count() == 0:
        create_chunks_files()
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        if os.path.exists(CHUNKS_DIR):
            chunk_files = sorted(os.listdir(CHUNKS_DIR))
            for file_name in chunk_files:
                if file_name.endswith(".txt"):
                    file_path = os.path.join(CHUNKS_DIR, file_name)
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        source = lines[0].replace("Source:", "").strip()
                        chunk_id = lines[1].replace("ID:", "").strip()
                        content = "".join(lines[4:]).strip()

                        vector = get_embedding(content)
                        ids.append(chunk_id)
                        documents.append(content)
                        metadatas.append({"source": source})
                        embeddings.append(vector)

        if ids:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        print(f"[+] Збережено {collection.count()} об'єктів у векторну базу ChromaDB!\n")
    return collection

def classify_rag(collection, text: str) -> str:
    query_vec = get_embedding(text)
    results = collection.query(query_embeddings=[query_vec], n_results=1)
    
    context = results['documents'][0][0] if results and results['documents'] else ""
    
    prompt = f"""Використовуючи наведений контекст з бази знань, визнач категорію тексту.
Можливі категорії: програмування, штучний інтелект, бази даних, комп'ютерні мережі.
Поверни ЛИШЕ назву категорії.

Контекст:
{context}

Текст:
{text}"""
    
    response = client.models.generate_content(
        model=MODEL_NAME, 
        contents=prompt
    )
    return str(response.text).strip().lower()

if __name__ == "__main__":
    collection = load_chunks_to_chroma()
    print("2: RAG СИСТЕМА \n")
    correct = 0
    
    for idx, item in enumerate(TEST_DATA, 1):
        pred = classify_rag(collection, item['text'])
        is_correct = (pred == item['label'])
        if is_correct:
            correct += 1
        print(f"#{idx}  Результат: [{pred}] - Правильна відповідь: [{item['label']}] ")
        
    accuracy = (correct / len(TEST_DATA)) * 100
    print(f"\nПідсумок: {correct}/{len(TEST_DATA)} вірно | Точність (Accuracy) = {accuracy:.1f}%")