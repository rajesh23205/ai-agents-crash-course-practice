import platform

from agents import function_tool


@function_tool
def get_python_version() -> str:
    print("get_python_version function has been called.")
    # platform.python_version() returns a clean string like "3.13.16"
    version_number = platform.python_version()

    # Format it with the word "Python" prepended
    return f"Python {version_number}"


if __name__ == "__main__":
    version = get_python_version()
    print(version)
