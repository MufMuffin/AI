from google import genai

client = genai.Client(
    api_key=""
)

article_url = "https://uk.wikipedia.org/wiki/Штучний_інтелект"

tasks = {
    "Варіант 1. Звичайний prompt": (
        "Проаналізуй статтю за цим посиланням та визнач її основні ідеї:\n\n"
        f"URL: {article_url}"
    ),
    "Варіант 2. Просунутий prompt": (
        "Ти викладач університету. Проаналізуй статтю за вказаним посиланням.\n"
        "Виконай завдання поетапно:\n"
        "1. Визнач основну тему.\n"
        "2. Виділи 3-5 ключових ідей.\n"
        "3. Визнач основні поняття.\n"
        "4. Сформулюй короткий висновок.\n"
        "Результат подай структуровано.\n\n"
        f"URL статті: {article_url}"
    ),
}

for task_name, prompt_text in tasks.items():
  print(f"{task_name}")
  print(f"PROMPT:\n{prompt_text}\n")

  interaction = client.interactions.create(
      model="gemini-3.5-flash-lite", 
      input=prompt_text
  )

  print("Відповідь:")
  print(interaction.output_text)