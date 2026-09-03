import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from config import ORDER_AGENT_MODEL

load_dotenv(find_dotenv())

from langchain_core.tools import tool
from langchain.agents import create_agent
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


ORDERS = {
    "ORD-001": {"status": "Shipped", "eta": "17/08/2026"},
    "ORD-002": {"status": "Delivered", "eta": "15/08/2026"},
    "ORD-003": {"status": "Processing", "eta": "25/08/2026"}
}

REFUND_POLICIES = {
    "electronics": "Electronics can be returned within 10 days, unopened",
    "clothing": "Clothing can be returned within 20 days, with tags attached",
    "food": "Food items are not refundable for safety reasons"
}


# Tools
@tool
def get_order_status(ord_id: str) -> str:
    """ Get orders from back end"""
    order = ORDERS.get(ord_id.upper())
    if not order:
        return f"No order found with ID {ord_id}"
    return f"Order {ord_id} is on status {order['status']} & ETA is {order['eta']}"

@tool
def get_refund_policy(category: str) -> str:
    """ Get refund info for product type"""
    policy = REFUND_POLICIES.get(category.lower())
    if not policy:
        return f"No refund policy for {category}"
    return policy

# Agent code
llm = init_chat_model(
    model=ORDER_AGENT_MODEL,
    temperature=0
)

deepeval_callback = CallbackHandler()

agent = create_agent(
    model = llm,
    tools=[get_order_status, get_refund_policy],
    system_prompt=(
          "You are a friendly customer-support agent. "
          "Use the available tools to answer order and refund questions. "
          "Keep replies short and helpful."
    )
)


def support_agent(user_input: str) -> str:
    """ Run the agent and return a response """
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": user_input
        }]
    },
    config={"callbacks": [deepeval_callback]})
    response = result["messages"][-1].content
    update_current_trace(output = response)
    return response

if __name__ == "__main__":
    print(support_agent("Where is my Order ORD-001?"))