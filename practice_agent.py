import asyncio
import os

import dotenv

# from agents import Agent, ModelSettings, Runner, trace
from agents import Agent, ModelSettings, RunContextWrapper, Runner, SQLiteSession, trace
from openai import OpenAI

from knowledge_reader import read_knowledge
from structured_output import MathAnswer
from user_context import Student

dotenv.load_dotenv()

session = SQLiteSession(
    session_id="python_learning_assistant",
    db_path="agent_memory.db",
)

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

    Keep your answers concise, clear, and beginner-friendly when needed.

    The student's name is {context.context.name}.
    The student's experience level is {context.context.experience_level}.
    The student's favorite language is {context.context.favorite_language}.

    Adjust your explanations according to the student's experience level.

    When you use the read_knowledge tool:
    - Explain the retrieved information in complete sentences.
    - Use the retrieved information to answer the question accurately.
    - Explain the concept clearly according to the student's experience level.
    - Do not answer with only a single word when the retrieved information
      supports a fuller explanation.
    - If the retrieved information does not answer the question,
      clearly say that the available knowledge is insufficient.
    """


# Create a python tutor agent
python_tutor_agent = Agent(
    name="Python Tutor",
    instructions=python_tutor_instructions,
    tools=[read_knowledge],
    # model_settings=ModelSettings(tool_choice="read_knowledge"),
)

# Ceate a Math agent
math_agent = Agent(
    name="Math Tutor",
    instructions="""
    You are a math expert.
    Solve mathematical questions clearly and concisely.
    """,
    output_type=MathAnswer,
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="""
    You are a routing agent. Your primary job is to hand off questions
    to the appropriate specialist.

    - For every question about Python, programming, Python syntax,
    or Python data structures, hand off to the Python Tutor.
    - For every mathematical question or calculation, hand off to
    the Math Tutor.
    - Do not answer these questions yourself.
    - If a question is clearly about Python, do not send it to the
    Math Tutor.
    """,
    handoffs=[python_tutor_agent, math_agent],
)

knowledge_tutor_agent = Agent(
    name="Knowledge Tutor",
    instructions="""
    Answer questions using the information returned by read_knowledge.

    - Explain retrieved information in complete sentences.
    - If the retrieved information is insufficient, say so clearly.
    - Do not use your general knowledge to fill gaps in the knowledge file.
    """,
    tools=[read_knowledge],
    model_settings=ModelSettings(tool_choice="read_knowledge"),
)


# Execute the agent


async def main():

    # # Test 1: Math Tutor + Structured Output
    # print("\n=== Test 1: Math Question ===")

    # with trace("Math question"):
    #     result = await Runner.run(
    #         triage_agent,
    #         "What is the area of a circle with a radius of 5 cm?",
    #         session=session,
    #     )

    # print("Final agent:", result.last_agent.name)
    # print("Output type:", type(result.final_output).__name__)

    # if isinstance(result.final_output, MathAnswer):
    #     print("Answer:", result.final_output.answer)
    #     print("Unit:", result.final_output.unit)
    #     print("Formula:", result.final_output.formula)
    # else:
    #     print(result.final_output)

    # print("=============================")

    # # Test 2: Python Tutor
    # print("\n=== Test 2: Python Question ===")

    # with trace("Python question"):
    #     result = await Runner.run(
    #         triage_agent,
    #         "What is the difference between a Python list and a tuple?",
    #         context=student_2,
    #         session=session,
    #     )

    # print("Final agent:", result.last_agent.name)
    # print("Output type:", type(result.final_output).__name__)
    # print(result.final_output)

    # print("=============================")

    # Test 3: Knowledge Retrieval
    print("\n=== Test 3: Knowledge Retrieval ===")

    with trace("Knowledge retrieval question"):
        result = await Runner.run(
            knowledge_tutor_agent,
            "What is Japan's population?",
            context=student_1,
            session=SQLiteSession("knowledge_retrieval_test"),
        )

    print("Final agent:", result.last_agent.name)
    print("Output type:", type(result.final_output).__name__)
    print(result.final_output)

    print("=============================")


if __name__ == "__main__":
    asyncio.run(main())
