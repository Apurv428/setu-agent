"""
scratch_agent.py — framework-free tool-calling agent loop.

No LangChain, no LangGraph. A strict JSON protocol:
  {"tool": "<name>", "args": {...}}  -> call a tool
  {"final": "<answer>"}              -> done
"""

# TODO: implement after mcp_server.py is verified via inspector
