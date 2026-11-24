import os
import time
from google import genai
from google.genai.errors import ClientError

API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-3.5-flash-lite"

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

def safe_generate(prompt_text: str) -> str:
    """Функція з затримкою та автоматичною паузою при отриманні помилки 429."""
    while True:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, 
                contents=prompt_text
            )
            time.sleep(2)  
            return str(response.text).strip().lower()
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("\n[!] Перевищено ліміт запитів (429). Очікування 20 секунд...")
                time.sleep(20)
            else:
                raise e

def classify_few_shot(text: str) -> str:
    prompt = f"""Визнач категорію наведеного тексту.
Можливі категорії: програмування, штучний інтелект, бази даних, комп'ютерні
мережі.
Поверни лише назву категорії за аналогією до наведених прикладів:

Текст: Написання коду на C++ вимагає ручного керування пам'яттю.
Категорія: програмування

Текст: Навчання алгоритму регресії на великому датасеті.
Категорія: штучний інтелект

Текст: Створення первинних ключів у таблицях PostgreSQL.
Категорія: бази даних

Текст: {text}
Категорія:"""
    
    return safe_generate(prompt)

if __name__ == "__main__":
    print("3: ІМІТАЦІЯ FINE-TUNING\n")
    correct = 0
    
    for idx, item in enumerate(TEST_DATA, 1):
        pred = classify_few_shot(item['text'])
        is_correct = (pred == item['label'])
        if is_correct:
            correct += 1
        print(f"#{idx:02d} Результат: [{pred}] - Правильна відповідь: [{item['label']}] ")
        
    accuracy = (correct / len(TEST_DATA)) * 100
    print(f"\nТочність (Accuracy): {correct}/{len(TEST_DATA)} ({accuracy:.1f}%)")