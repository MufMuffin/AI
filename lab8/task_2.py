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

def single_agent(task: str, text: str) -> str:
    return ask_gemini(f"{task}\n\n{text}")

def multi_agent(task: str, text: str) -> None:

    plan_text = ask_gemini(
        f"Завдання: {task}\nТекст:\n{text}",
        system_prompt="Ти планувальник. Склади короткий покроковий план аналізу тексту (кожен пункт з нового рядка)."
    )
    print("\n[АГЕНТ 1 — ПЛАНУВАЛЬНИК (План)]:")
    print(plan_text)

    subtasks = [line.strip() for line in plan_text.splitlines() if line.strip()]
    results = []
    for step in subtasks:
        ans = ask_gemini(
            f"Підзадача: {step}\nТекст:\n{text}",
            system_prompt="Ти виконавець. Виконай підзадачу строго за наданим текстом без вигадок."
        )
        results.append(f"{step}\nВідповідь: {ans}")

    draft = "\n\n".join(results)
    print("\n[АГЕНТ 2 — ВИКОНАВЕЦЬ (Виконання плану)]:")
    print(draft)

    final_answer = ask_gemini(
            f"Завдання користувача: {task}\n\nВихідний текст:\n{text}\n\nРезультати виконання пунктів:\n{draft}",
            system_prompt=(
                "Ти перевіряльник і редактор. Перевір результати виконавця на точність за вихідним текстом, "
                "об'єднай усі зібрані дані в єдину зв'язну фінальну відповідь на поставлене завдання користувача. "
                "Не дублюй список підзадач і не розписуй відповідь по пунктах плану — дай цілісний підсумковий текст."
            )
        )
    print("\n[АГЕНТ 3 — ПЕРЕВІРЯЛЬНИК (Фінальна відповідь)]:")
    print(final_answer)


if __name__ == "__main__":
    text = Path("text1.txt").read_text(encoding="utf-8")
    task = "Проаналізуй текст: визнач тему, ключові ідеї та зроби короткий висновок."

    print("ВАРІАНТ 1: ОДИН АГЕНТ\n")
    print(single_agent(task, text))

    print("\nВАРІАНТ 2: МУЛЬТИАГЕНТНА СИСТЕМА\n")
    multi_agent(task, text)