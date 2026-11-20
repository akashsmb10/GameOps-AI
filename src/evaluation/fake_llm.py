"""A scripted stand-in for a tool-calling chat model, used to test the LLM path of the agent
graph (tool selection by the model, tool execution, citation checking) without an LLM server.
It verifies the wiring only - it says nothing about the quality of a real model."""

from __future__ import annotations

from langchain_core.messages import AIMessage


class ScriptedToolModel:
    def __init__(self, script: list[AIMessage]):
        self.script, self.calls = list(script), 0

    def bind_tools(self, tools):
        self.tools = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.calls += 1
        return self.script.pop(0) if self.script else AIMessage(content="No further answer.")


def tool_call(name: str, args: dict, i: int = 0) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{i}"}])
