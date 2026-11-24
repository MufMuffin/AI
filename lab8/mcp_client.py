import asyncio
import sys
from pathlib import Path
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

client = genai.Client(api_key="")
MODEL_ID = "gemini-3.5-flash-lite"


async def main():
    server_path = str(Path(__file__).parent / "mcp_server.py")
    server_params = StdioServerParameters(command=sys.executable, args=[server_path])

    task = (
        "Прочитай файл text1.txt і проаналізуй текст: визнач тему, ключові ідеї та зроби короткий висновок. "
        "А також обчисли за допомогою калькулятора: (5 * 5) + 5."
    )
    print(f"1. ЗАПИТ КОРИСТУВАЧА:\n{task}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            tools = [
                {
                    "function_declarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema,
                        }
                        for t in mcp_tools.tools
                    ]
                }
            ]

            print("2. ІНСТРУМЕНТИ MCP SERVER:")
            for t in mcp_tools.tools:
                print(f" - {t.name}")

            chat_history = [task]
            response = await client.aio.models.generate_content(
                model=MODEL_ID,
                contents=chat_history,
                config=types.GenerateContentConfig(tools=tools),
            )

            if response.function_calls:
                print("\n3. ВИКЛИК ІНСТРУМЕНТІВ ЧЕРЕЗ MCP:")
                chat_history.append(response.candidates[0].content)

                for call in response.function_calls:
                    print(f" -> {call.name}({call.args})")
                    result = await session.call_tool(call.name, call.args)
                    tool_output = result.content[0].text
                    print(f" <- Результат: {tool_output}")

                    chat_history.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": tool_output},
                        )
                    )

                response = await client.aio.models.generate_content(
                    model=MODEL_ID,
                    contents=chat_history,
                    config=types.GenerateContentConfig(tools=tools),
                )

            print(f"\n4. ВІДПОВІДЬ МОДЕЛІ:\n{response.text}")


if __name__ == "__main__":
    asyncio.run(main())