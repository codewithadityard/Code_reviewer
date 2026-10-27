import asyncio 
import re 
import json
from planner_agent import get_planner_node 
from coding_agent import get_code_node 
from testing_agent import get_qa_node 
from review_agent import get_review_node 
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage

import asyncio

from planner_agent import get_planner_node
# from coding_agent import get_code_node
# from testing_agent import get_qa_node
# from review_agent import get_review_node

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage


async def debug_team(team, task):
    print("=" * 100)
    print("TASK:")
    print(task)
    print("=" * 100)

    async for event in team.run_stream(task=task):

        print("\n" + "=" * 80)
        print("EVENT TYPE :", type(event))
        print("CLASS      :", event.__class__.__name__)

        if isinstance(event, TextMessage):
            print("\n---- TEXT MESSAGE ----")
            print("SOURCE :", event.source)
            print("CONTENT:")
            print(repr(event.content))
            print("----------------------")

        elif isinstance(event, TaskResult):
            print("\n**** TASK RESULT ****")
            print("STOP REASON :", event.stop_reason)
            print("TOTAL MESSAGES :", len(event.messages))

            for i, msg in enumerate(event.messages):
                print("\n" + "-" * 60)
                print(f"MESSAGE #{i}")
                print("TYPE   :", type(msg))
                print("CLASS  :", msg.__class__.__name__)

                if hasattr(msg, "source"):
                    print("SOURCE :", msg.source)

                if hasattr(msg, "content"):
                    print("CONTENT:")
                    print(repr(msg.content))

            print("*********************")

        else:
            print("\n---- OTHER EVENT ----")
            print(event)
            print("---------------------")


async def main():

    planner = get_planner_node()

    await debug_team(
        planner,
        "Write Python code to generate Fibonacci series for n numbers."
    )

    # Uncomment ONE at a time.

    coder = get_code_node()
    await debug_team(
        coder,
       "Write Python code to generate Fibonacci series for n numbers."
    )

    qa = get_qa_node()
    await debug_team(
        qa,
         "Test this code:\n```python\nprint('Hello')\n```"
     )

    reviewer = get_review_node()
    await debug_team(
         reviewer,
         "Review this code:\n```python\nprint('Hello')\n```"
     )


if __name__ == "__main__":
    asyncio.run(main())