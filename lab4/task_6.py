from google import genai
from google.genai import types

client = genai.Client(
    api_key=""
)

task_prompt = (
    "Згенеруй 3 унікальні та креативні назви для мобільного додатка, "
    "який допомагає людям вивчати іноземні мови за допомогою штучного інтелекту."
)

print("ДОСЛІДЖЕННЯ ПАРАМЕТРА TEMPERATURE")


temperatures = [0, 0.5, 1.0]

for temp in temperatures:
  config = types.GenerateContentConfig(temperature=temp)
  response = client.models.generate_content(
      model="gemini-3.5-flash-lite", contents=task_prompt, config=config
  )
  print(f"\n--- Temperature = {temp} ---")
  print(response.text)

print("ДОСЛІДЖЕННЯ ПАРАМЕТРА TOP-P")


top_p_values = [0.5, 0.8, 1.0]

for tp in top_p_values:
  config = types.GenerateContentConfig(top_p=tp)
  response = client.models.generate_content(
      model="gemini-3.5-flash-lite", contents=task_prompt, config=config
  )
  print(f"\n--- Top-P = {tp} ---")
  print(response.text)