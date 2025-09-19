import weave
import uuid
import art
from art.langgraph import init_chat_model

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dataclasses import asdict
from textwrap import dedent


from tasks.email.model import *
from tasks.email.functions import *
from utils.judgement_llm import judge_correctness




@weave.op
async def rollout(
    model: art.Model, 
    task_scenario,
    MAX_TURNS = 20
) -> ProjectTrajectory:
    scenario = task_scenario.scenario

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id,
            "step": task_scenario.step,
        },
    )

    system_prompt = dedent(
        f"""
        You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {MAX_TURNS} turns to find the answer, so if your first search doesn't find the answer, you can try with different keywords.

        User's email address is {scenario.inbox_address}
        Today's date is {scenario.query_date}

        When you have found the answer, use the return_final_answer_tool to provide your final answer along with the source message IDs.
        """
    )

    # Store final answer in trajectory
    final_answer = None

    # Define tools inside the rollout function to access local variables
    @tool
    def search_inbox_tool(keywords: list[str]) -> list[dict]:
        """Search the inbox for emails matching the given keywords and return
        a list of dictionaries so the LLM can easily consume them."""
        results = search_emails(
            inbox=scenario.inbox_address,
            keywords=keywords,
            sent_before=scenario.query_date,
        )
        return [asdict(result) for result in results]

    @tool
    def read_email_tool(message_id: str) -> dict | None:
        """Read a specific email by message ID."""
        email = read_email(message_id)
        if email:
            return email.model_dump()
        return None

    @tool
    def return_final_answer_tool(answer: str, reference_message_ids: list[str]) -> dict:
        """Return the final answer and the message IDs of the emails that were used to generate the answer."""
        nonlocal final_answer
        final_answer = FinalAnswer(answer=answer, source_ids=reference_message_ids)
        return final_answer.model_dump()

    # Create LangGraph tools
    tools = [search_inbox_tool, read_email_tool, return_final_answer_tool]

    chat_model = init_chat_model(model.name, temperature=1.0)

    # Create the LangGraph ReAct agent
    react_agent = create_react_agent(chat_model, tools)
    # react_agent = create_agent(chat_model, tools)
    

    try:
        # Run the agent
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": MAX_TURNS,
        }

        await react_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=scenario.question),
                ]
            },
            config=config,
        )

        # Check if we got a final answer
        if final_answer:
            traj.final_answer = final_answer
            # Score the trajectory
            correctness_judge_response = await judge_correctness(
                scenario, traj.final_answer.answer
            )
            traj.metrics["correct"] = float(correctness_judge_response.accept)

    except Exception as e:
        print(f"Error running LangGraph agent: {e}")
        # Add error information to trajectory
        traj.messages_and_choices.append(
            {"role": "assistant", "content": f"Error: {str(e)}"}
        )

    return traj