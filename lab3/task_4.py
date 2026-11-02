from google import genai

client = genai.Client(api_key="API-KEY")

user_task = "Поясни, чому важливо правильно обробляти винятки (Exceptions / try-catch) у коді."

roles = {
    "1. Викладач": "Ти — досвідчений та терплячий викладач програмування в університеті. Твоя мета — пояснити матеріал методично, зрозуміло для студентів, використовуючи аналогії та навчальний тон. Обсяг: до 4-5 речень.",
    
    "2. Програміст-практик": "Ти — Senior Software Engineer з 10-річним досвідом розробки високонавантажених систем. Поясни тему з точки зору продакшену, надійності системи, логування та підтримки коду. Використовуй професійний сленг. Обсяг: до 4-5 речень.",
    
    "3. Науковець": "Ти — науковець у галузі програмної інженерії та теорії систем. Поясни тему строгою академічною мовою, спираючись на поняття відмовостійкості, надійності ПЗ та теорії систем. Обсяг: до 4-5 речень."
}

print(f"Завдання: {user_task}")

for role_name, system_instruction_text in roles.items():
    print(f"\nРоль: {role_name}")
    
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        system_instruction=system_instruction_text,
        input=user_task
    )
    
    print("Відповідь:")
    print(interaction.output_text)