import weave
import uuid
import art
from art.langgraph import init_chat_model

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dataclasses import asdict
from textwrap import dedent

from tasks.email.model import *
from tasks.email.functions import *
from utils.judgement_llm import judge_correctness

MAX_TURNS = 20

@weave.op
async def rollout(
    model: art.Model, 
    task_scenario: EmailScenario,  # 타입 힌트 명확화
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
        try:
            results = search_emails(
                inbox=scenario.inbox_address,
                keywords=keywords,
                sent_before=scenario.query_date,
            )
            return [asdict(result) for result in results]
        except Exception as e:
            print(f"Error in search_inbox_tool: {e}")
            return []

    @tool
    def read_email_tool(message_id: str) -> dict | None:
        """Read a specific email by message ID."""
        try:
            email = read_email(message_id)
            if email:
                return email.model_dump()
            return None
        except Exception as e:
            print(f"Error in read_email_tool: {e}")
            return None

    @tool
    def return_final_answer_tool(answer: str, reference_message_ids: list[str]) -> dict:
        """Return the final answer and the message IDs of the emails that were used to generate the answer."""
        nonlocal final_answer
        final_answer = FinalAnswer(answer=answer, source_ids=reference_message_ids)
        return final_answer.model_dump()

    # Create LangGraph tools
    tools = [search_inbox_tool, read_email_tool, return_final_answer_tool]

    try:
        chat_model = init_chat_model(model.name, temperature=1.0)
        
        # Create the LangGraph ReAct agent
        react_agent = create_react_agent(chat_model, tools)

        # Run the agent
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": MAX_TURNS,
        }

        agent_response = await react_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=scenario.question),
                ]
            },
            config=config,
        )
        
        # Extract messages from agent response for trajectory
        if "messages" in agent_response:
            for msg in agent_response["messages"]:
                if hasattr(msg, 'dict'):
                    traj.messages_and_choices.append(msg.dict())
                elif isinstance(msg, dict):
                    traj.messages_and_choices.append(msg)
                else:
                    # Convert message to dict format
                    msg_dict = {
                        "role": getattr(msg, 'type', 'unknown'),
                        "content": getattr(msg, 'content', ''),
                    }
                    if hasattr(msg, 'tool_calls'):
                        msg_dict["tool_calls"] = msg.tool_calls
                    traj.messages_and_choices.append(msg_dict)

        # Check if we got a final answer
        if final_answer:
            traj.final_answer = final_answer
            # Score the trajectory
            try:
                correctness_judge_response = await judge_correctness(
                    scenario, traj.final_answer.answer
                )
                traj.metrics["correct"] = float(correctness_judge_response.accept)
            except Exception as e:
                print(f"Error in correctness judging: {e}")
                traj.metrics["correct"] = 0.0

    except Exception as e:
        print(f"Error running LangGraph agent: {e}")
        # Add error information to trajectory
        traj.messages_and_choices.append(
            {"role": "assistant", "content": f"Error: {str(e)}"}
        )
        traj.metrics["error"] = 1.0

    return traj