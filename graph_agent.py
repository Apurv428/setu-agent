"""
graph_agent.py — LangGraph ReAct agent consuming the sarvam-tools MCP server.

Contrast with scratch_agent.py:
  - The framework (LangGraph) manages the tool-call loop and state.
  - sarvam-30b's native tool calling replaces the hand-written JSON protocol.
  - The MCP server is the same one used by the scratch agent.
  - MemorySaver checkpointer keyed by thread_id enables multi-turn memory.

Run:  python graph_agent.py [optional query text]
"""

import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Module-level checkpointer so memory persists across run() calls in the same process
_checkpointer = MemorySaver()


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="sarvam-30b",
        base_url="https://api.sarvam.ai/v1",
        api_key=os.environ["SARVAM_API_KEY"],
    )


async def run(user_input: str, thread_id: str = "default") -> str:
    """
    Run the LangGraph ReAct agent.

    user_input: a text query or audio file path (passed straight to the agent).
    thread_id: conversation thread identifier; reuse the same id across calls
               to maintain memory across turns.
    Returns the final answer string.
    """
    async with MultiServerMCPClient({
        "sarvam": {
            "command": "python",
            "args":    ["mcp_server.py"],
            "transport": "stdio",
        }
    }) as mcp_client:
        tools  = mcp_client.get_tools()
        agent  = create_react_agent(_llm(), tools, checkpointer=_checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = await agent.ainvoke({"messages": [("user", user_input)]}, config)
        return result["messages"][-1].content


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What is the capital of India? Answer in Hindi and speak the reply."
    print(f"Query: {query}\n")
    answer = asyncio.run(run(query, thread_id=thread_id))
    print(f"Answer: {answer}")
