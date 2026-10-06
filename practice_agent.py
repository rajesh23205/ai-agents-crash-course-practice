import asyncio
import os

import dotenv

# from agents import Agent, ModelSettings, Runner, trace
from agents import Agent, RunContextWrapper, Runner, trace
from openai import OpenAI

from calculator import calculator
from python_version import get_python_version
from user_context import Student

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

# student object
student_1 = Student(
    name="Rajesh", experience_level="beginner", favorite_language="Python"
)

student_2 = Student(
    name="Roshan", experience_level="advanced", favorite_language="Python"
)


def python_tutor_instructions(context: RunContextWrapper[Student], agent: Agent) -> str:
    return f"""
    You are a Python expert. Provide tuition and clear all doubts
    about the Python programming language.

    Keep your answers concise.

    The student's name is {context.context.name}.
    The student's experience level is {context.context.experience_level}.
    The student's favorite language is {context.context.favorite_language}.

    Adjust your explanations according to the student's experience level.
    """


# Create a python tutor agent
python_tutor_agent = Agent(
    name="Python Tutor",
    instructions=python_tutor_instructions,
    tools=[calculator, get_python_version],
    # model_settings=ModelSettings(tool_choice="calculator"),
)

# Ceate a Math agent
math_agent = Agent(
    name="Math Tutor",
    instructions="""
    You are a math expert.
    Solve mathematical questions clearly and concisely.
    """,
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="""
    You are a routing agent.
    If the user asks a Python question, hand it to the Python Tutor.
    If the user asks a math question, hand it to the Math Tutor.
    """,
    handoffs=[python_tutor_agent, math_agent],
)


# Execute the agent
async def main():
    # with trace("Python experience tutor"):
    #     result = await Runner.run(
    #         python_tutor_agent,
    #         # "What is 25 multiplied by 48? use required function.",
    #         "What Python version am I using? use required function. Include my name as well in the response",
    #         # session=session,
    #         context=student_2,
    #     )

    # print(result.final_output)

    with trace("Agent trigger execution"):
        result = await Runner.run(
            triage_agent,
            "What is the area of a circle with a radius of 5 cm?",
        )

    print(result.final_output)

    with trace("Agent trigger execution"):
        result = await Runner.run(
            triage_agent,
            "What is the difference between a Python list and a tuple?",
            context=student_2,
        )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
