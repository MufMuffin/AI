from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Lab8-Demo-Server")
BASE_DIR = Path(__file__).parent


@mcp.tool()
def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(ch in allowed for ch in expression):
            return "Помилка: недопустимі символи у виразі"
        return str(eval(expression))
    except Exception as e:
        return f"Помилка обчислення: {e}"


@mcp.tool()
def read_local_file(filename: str) -> str:
    path = BASE_DIR / filename
    if not path.exists():
        return f"Файл '{filename}' не знайдено."
    return path.read_text()


if __name__ == "__main__":
    mcp.run()