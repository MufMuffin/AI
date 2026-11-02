from google import genai

client = genai.Client(api_key="API-KEY")

print("Простий PROMPT")
print("Розкажи про об'єктно-орієнтоване програмування.\n")

interaction_initial = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input="Розкажи про об'єктно-орієнтоване програмування."
)
print("Відповідь (Простий):")
print(interaction_initial.output_text, "\n")

improved_prompt = """
Ти — Senior Python-розробник і ментор. Поясни поняття «Об'єктно-орієнтоване програмування (ООП)» для студента, який тільки починає вивчати програмування. 
Структуруй відповідь у вигляді двох блоків: 
1. Суть ООП простими словами; 
2. Перелічи та стисло поясни 3 основні принципи (інкапсуляція, успадкування, поліморфізм). 
Використовуй дружній та зрозумілий стиль.
"""

print("Вдосконалений PROMPT ")
print(improved_prompt.strip(),"\n\n")

interaction_improved = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input=improved_prompt.strip()
)
print("Відповідь (Вдосконалений):")
print(interaction_improved.output_text)