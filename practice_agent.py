import asyncio
import os

import dotenv
from agents import Agent, Runner, trace
from openai import OpenAI

from calculator import calculator

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    print(
        """Error: OPENAI_API_KEY environment variable not set. Please copy the .env.template file as .env and fill it in.
    
    You can execute these commands in the terminal to get started:
    cp .env.template .env
    code .env
    """
    )

# Test OpenAI Access
print(
    OpenAI()
    .responses.create(
        model=os.environ["OPENAI_DEFAULT_MODEL"], input="Say: We are up and running!"
    )
    .output_text
)

# Create a python tutor agent
python_tutor_agent = Agent(
    name="Python Tutor",
    instructions="""
    You are a python expert, provide tution and clear all the doubts of python the coading language.
    You give concise answer
    """,
    tools=[calculator],
)


# Execute the agent
async def main():
    with trace("Python experience tutor"):
        result = await Runner.run(
            python_tutor_agent,
            "Use the calculator tool to calculate, What is 25 multiplied by 48?",
        )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
