import time
from pathlib import Path
from google import genai
from google.genai import types

client = genai.Client(api_key="")
MODEL_ID = "gemini-3.5-flash-lite"


def ask_gemini(prompt: str, system_prompt: str = None) -> str:
    config = types.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None
    for _ in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        except Exception:
            time.sleep(3)
    return "Помилка з'єднання з моделлю"


def run_multi_agent_system(task: str, text: str) -> None:
    print(f"1. ПОЧАТКОВИЙ ЗАПИТ:\n{task}\n")

    plan_text = ask_gemini(
        f"Завдання: {task}\nТекст:\n{text}",
        system_prompt="Ти планувальник. Склади короткий покроковий план виконання завдання (кожен пункт з нового рядка)."
    )
    print(f"2. ПЛАН АГЕНТА-ПЛАНУВАЛЬНИКА:\n{plan_text}\n")

    subtasks = [line.strip() for line in plan_text.splitlines() if line.strip()]
    results = []
    print("3. ВИКОНАННЯ ПІДЗАДАЧ (Агент-виконавець):")
    for step in subtasks:
        ans = ask_gemini(
            f"Підзадача: {step}\nТекст:\n{text}",
            system_prompt="Ти виконавець. Виконай підзадачу строго за наданим текстом без вигадок."
        )
        print(f"  -> Підзадача: {step}")
        print(f"     Результат виконавця: {ans}\n")
        results.append(f"{step}\nВідповідь: {ans}")

    draft = "\n\n".join(results)

    review_result = ask_gemini(
        f"Завдання користувача: {task}\n\nВихідний текст:\n{text}\n\nРезультати виконавця:\n{draft}",
        system_prompt=(
            "Ти перевіряльник. Звір результати виконавця з вихідним текстом. "
            "Коротко зафіксуй результат перевірки: чи всі факти точні, чи немає помилок або вигадок."
        )
    )
    print(f"4. РЕЗУЛЬТАТ ПЕРЕВІРКИ (Агент-перевіряльник):\n{review_result}\n")

    final_answer = ask_gemini(
        f"Завдання користувача: {task}\n\nВихідний текст:\n{text}\n\nРезультати виконавця:\n{draft}\n\nЗауваження перевіряльника:\n{review_result}",
        system_prompt=(
            "Ти фінальний редактор. На основі перевірених даних сформуй єдину зв'язну остаточну відповідь "
            "на завдання користувача. Не дублюй список підзадач — дай цілісний підсумковий текст."
        )
    )
    print(f"5. ФІНАЛЬНА ВІДПОВІДЬ:\n{final_answer}\n")


if __name__ == "__main__":
    test_cases = [
        ("text1.txt", "Визнач головну тему тексту, виділи 3 ключові ідеї та сформулюй короткий висновок."),
        ("text2.txt", "Знайди у тексті основні проблеми або виклики та запропоновані способи їх вирішення."),
        ("text3.txt", "Склади стислий переказ тексту для людини, яка не має часу читати оригінал (до 5 речень)."),
    ]

    for i, (file_name, user_task) in enumerate(test_cases, start=1):
        print(f"ТЕСТОВА ЗАДАЧА №{i} (Файл: {file_name})")
        source_text = Path(file_name).read_text(encoding="utf-8")
        run_multi_agent_system(user_task, source_text)