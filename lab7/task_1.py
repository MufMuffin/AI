import json
import math
import os
import sys
import time

from google import genai
from google.genai import errors, types

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-3.5-flash-lite"

client = genai.Client(api_key=API_KEY)

def ask_model(contents, config):
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
        except errors.APIError as e:
            if e.code in (429, 500, 503, 504):
                wait_sec = 10 * (attempt + 1)
                print(f"[{MODEL}] тимчасова помилка {e.code}, очікування {wait_sec} с (спроба {attempt + 1}/3)...")
                time.sleep(wait_sec)
            else:
                raise
    raise RuntimeError("Модель зараз недоступна.")


SAFE_ENV = {
    "__builtins__": {},
    "pi": math.pi,
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
}

def calculator(expression: str) -> str:
    try:
        clean_expr = expression.replace("^", "**")
        result = eval(clean_expr, SAFE_ENV, {})
        return str(result)
    except ZeroDivisionError:
        return "Помилка: ділення на нуль"
    except Exception as e:
        return f"Помилка: {e}"

calculator_declaration = {
    "name": "calculator",
    "description": (
        "Обчислює математичний вираз, наприклад (12+18+25+30)/4. "
        "Підтримує +, -, *, /, **, %, дужки, sqrt, abs, sin, cos та pi."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Математичний вираз"}
        },
        "required": ["expression"],
    },
}


history: list[types.Content] = []
MAX_MESSAGES = 20


def add_turn_to_history(question: str, answer: str):
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=answer)]))
    while len(history) > MAX_MESSAGES:
        history.pop(0)


PLAN_PROMPT = """Ти модуль планування AI-агента "Помічник студента".
Доступний інструмент: calculator (математичні обчислення).
Враховуй історію діалогу. Для останнього запиту користувача поверни ТІЛЬКИ JSON:
{"needs_tool": true або false, "plan": ["крок 1", "крок 2", ...]}
У плані має бути від 2 до 5 коротких кроків."""

SYSTEM_PROMPT = """Ти AI-агент "Помічник студента". Відповідай українською.
Можливості: пояснювати навчальні поняття, складати плани підготовки,
створювати тестові запитання та виконувати обчислення.
Усі математичні обчислення виконуй ТІЛЬКИ через інструмент calculator.
Якщо інструмент повернув помилку — чітко поясни її користувачу.
Враховуй попередні повідомлення діалогу."""


def make_plan(question: str) -> dict:
    contents = list(history) + [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    config = types.GenerateContentConfig(
        system_instruction=PLAN_PROMPT,
        response_mime_type="application/json",
    )
    response = ask_model(contents, config)
    try:
        data = json.loads(response.text)
        if isinstance(data.get("plan"), list) and data["plan"]:
            return data
    except Exception:
        pass
    return {"needs_tool": False, "plan": ["Проаналізувати запит", "Сформувати відповідь"]}


def run_agent(question: str, show_tools: bool = True):
    plan_data = make_plan(question)
    plan_steps = plan_data["plan"]

    config = types.GenerateContentConfig(
        system_instruction=f"{SYSTEM_PROMPT}\n\nПлан дій:\n" + "\n".join(f"- {step}" for step in plan_steps),
        tools=[types.Tool(function_declarations=[calculator_declaration])],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents = list(history) + [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    used_tools = []
    answer = "Не вдалося виконати завдання."

    for _ in range(5):
        response = ask_model(contents, config)
        calls = response.function_calls
        if not calls:
            answer = (response.text or "").strip() or answer
            break

        contents.append(response.candidates[0].content)
        tool_responses = []

        for call in calls:
            expr = call.args.get("expression", "")
            calc_result = calculator(expr)
            used_tools.append(f"{expr} = {calc_result}")
            if show_tools:
                print(f"  [calculator] {expr} -> {calc_result}")
            tool_responses.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": calc_result},
                )
            )

        contents.append(types.Content(role="user", parts=tool_responses))

    add_turn_to_history(question, answer)
    return plan_steps, used_tools, answer


TESTS = [
    "Що таке змінна у програмуванні?",
    "Обчисли середнє арифметичне чисел 12, 18, 25, 30 та поясни результат.",
    "Порахуй площу кола з радіусом 7 і площу квадрата зі стороною 7 та скажи, яка з них більша.",
    "Порадь мені завдання для практики.",
    "Обчисли 10 / 0.",
]


def run_tests():
    print("=== 1. Перевірка пам'яті (3 послідовні повідомлення) ===")
    history.clear()
    for q in ["Я вивчаю Python.", "Порадь мені завдання для практики.", "Поясни, що таке цикл for."]:
        _, _, ans = run_agent(q, show_tools=False)
        print(f"Користувач: {q}\nАгент: {ans}\n")

    print("=== 2. Тестові сценарії (5 завдань) ===")
    for idx, q in enumerate(TESTS, 1):
        history.clear()
        if idx == 4:
            run_agent("Я вивчаю Python.", show_tools=False)

        plan_steps, tools, ans = run_agent(q, show_tools=False)
        print(f"\nТест {idx}: {q}")
        print("План:", " -> ".join(plan_steps))
        print("Інструмент:", ", ".join(tools) if tools else "-")
        print("Відповідь:", ans)


def main():
    print("AI-агент 'Помічник студента' готовий (exit — вихід, /reset — очистити пам'ять).")
    while True:
        question = input("\nВи: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        if question == "/reset":
            history.clear()
            print("Історію діалогу очищено.")
            continue
        _, _, answer = run_agent(question, show_tools=True)
        print(f"\nАгент: {answer}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        main()