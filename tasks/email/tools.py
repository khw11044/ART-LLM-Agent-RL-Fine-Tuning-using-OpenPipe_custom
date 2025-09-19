


from dataclasses import asdict

from langchain_core.tools import tool
from tasks.email.model import *
from tasks.email.functions import *






# Define tools inside the rollout function to access local variables
@tool
def search_inbox_tool(keywords: list[str], scenario) -> list[dict]:
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
def return_final_answer_tool(final_answer: str, answer: str, reference_message_ids: list[str]) -> dict:
    """Return the final answer and the message IDs of the emails that were used to generate the answer."""
    final_answer = FinalAnswer(answer=answer, source_ids=reference_message_ids)
    return final_answer.model_dump()

