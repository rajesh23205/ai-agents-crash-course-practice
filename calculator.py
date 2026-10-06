from agents import function_tool


@function_tool
def calculator(a: float, b: float, operation: str) -> float | str:
    print("function has been called")
    match operation:
        case "add":
            return a + b
        case "subtract":
            return a - b
        case "multiply":
            return a * b
        case "divide":
            if b == 0:
                return "Cannot divide by zero"
            return a / b
        case _:
            return "Please provide a valid operation"


if __name__ == "__main__":
    result = calculator(5, 20, "add")
    print(result)
