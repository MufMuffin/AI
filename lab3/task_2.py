from google import genai

client = genai.Client(api_key="API-KEY")

zero_shot_input = f"""
Визнач мову програмування для наступного фрагмента коду та коротко опиши його призначення: 
print(list(map(lambda x: x**2, range(5))))
"""

print("ZERO-SHOT PROMPT")
print(zero_shot_input.strip(), "\n\n")

interaction_zero = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input=zero_shot_input.strip()
)
print("Відповідь (Zero-shot):")
print(interaction_zero.output_text, "\n")

few_shot_input = f"""
Визнач мову програмування для фрагмента коду та опиши його призначення у такому ж форматі, як у прикладах нижче.

Приклад 1:
Код: console.log("Hello, World!");
Мова: JavaScript
Опис: Виводить текстове повідомлення у консоль.

Приклад 2:
Код: System.out.println("Test");
Мова: Java
Опис: Здійснює виведення рядка тексту в стандартний потік виведення.

Основне завдання:
Код: print(list(map(lambda x: x**2, range(5))))
Мова:
"""

print("FEW-SHOT PROMPT")
print(few_shot_input.strip(), "\n\n")

interaction_few = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input=few_shot_input.strip()
)
print("Відповідь (Few-shot):")
print(interaction_few.output_text)
