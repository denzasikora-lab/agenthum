"""AgentHum: a compact OpenAI-compatible human-in-the-loop coding agent.

One file contains the full loop: ask the LLM, run requested tools, append tool
results, and continue until the model returns a final message.
"""

from collections.abc import Callable
import json
import subprocess
import sys
from typing import Any

import requests


# LLM endpoint settings. Local servers often accept any API key placeholder.
LLM_BASE_URL = "http://ip:port/v1"
LLM_API_KEY = "..."
LLM_MODEL = "..."
LLM_HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"}

# Runtime limits.
MAX_TURNS = 1000
COMMAND_TIMEOUT_SECONDS = 120
TOOL_RESULT_PREVIEW_CHARS = 500

# Operating policy for the model.
SYSTEM_PROMPT = """\
You are a coding agent. Your job is to help the user with programming tasks.

You have access to two tools:
- `bash` executes shell commands and returns stdout/stderr.
- `ask_user` asks the human for clarification when required information is missing.

Workflow:
1. Plan what needs to be done.
2. Use `bash` to inspect files, run commands, and make local changes.
3. Use `ask_user` only when the task is genuinely blocked by missing context or a decision the human must make.
4. After gathering enough information or completing the task, give your final answer in natural language.
5. To finish, reply with a regular message (no tool call).

Be concise. Explain what you're doing before each command."""

# Tool schemas shown to the model; implementations are mapped below.
LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the human operator a short clarification question and return their answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A concise question that is necessary to continue the task.",
                    }
                },
                "required": ["question"],
            },
        },
    },
]


def run_bash(command: str) -> str:
    """Run a shell command and return stdout, stderr, and the exit code."""
    try:
        command_result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        out = command_result.stdout + (f"\nSTDERR:\n{command_result.stderr}" if command_result.stderr else "")
        return f"Exit code: {command_result.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {COMMAND_TIMEOUT_SECONDS}s"


def ask_user(question: str) -> str:
    """Ask the person running the agent for missing context."""
    print(f"\n❓ Agent asks: {question}")
    answer = input("Your answer: ").strip()
    return answer or "(no answer provided)"


# Model-facing tool names mapped to local functions.
TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "bash": run_bash,
    "ask_user": ask_user,
}


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a model-requested tool call to a local Python function."""
    func = TOOL_HANDLERS.get(name)
    if not func:
        return f"Error: unknown tool '{name}'"
    try:
        return func(**arguments)
    except Exception as e:
        return f"Error calling {name}: {e}"


def call_llm(messages: list[dict[str, object]]) -> tuple[str, list[dict[str, Any]]]:
    """Send the current conversation to the configured chat completions API."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "tools": LLM_TOOLS,
        "tool_choice": "auto",
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    llm_http_response = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, headers=LLM_HEADERS)
    llm_http_response.raise_for_status()
    msg = llm_http_response.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    tool_calls = msg.get("tool_calls") or []
    return content, tool_calls


def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """Parse tool-call JSON and return a safe error-shaped payload on failure."""
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid tool arguments JSON: {e}"}
    if not isinstance(parsed, dict):
        return {"_error": "Tool arguments must decode to a JSON object."}
    return parsed


def format_tool_preview(result: str) -> str:
    """Keep terminal logs readable while preserving the full result for the LLM."""
    preview = result[:TOOL_RESULT_PREVIEW_CHARS]
    suffix = "..." if len(result) > TOOL_RESULT_PREVIEW_CHARS else ""
    return f"{preview}{suffix}"


def agent_loop(user_message: str) -> None:
    """Run the agent until the model stops asking for tool calls."""
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}
    ]
    for turn in range(1, MAX_TURNS + 1):
        print(f"\n{'='*60}\n🔄 Turn {turn}\n{'='*60}")
        content, tool_calls = call_llm(messages)
        if content:
            print(f"\n🤖 {content}")
        if not tool_calls:
            print("(no text output)" if not content else "")
            print("✅ Agent finished")
            return

        # One assistant tool_calls message, then one result per tool_call_id.
        messages.append({"role": "assistant", "content": content or None, "tool_calls": tool_calls})

        prefix = "\n" if content else ""
        for tool_call in tool_calls:
            function = tool_call["function"]["name"]
            arguments = parse_tool_arguments(tool_call["function"].get("arguments", "{}"))
            tool_call_id = tool_call["id"]
            print(f"{prefix}🔧 Tool: {function}({json.dumps(arguments, ensure_ascii=False)})")
            result = arguments["_error"] if "_error" in arguments else call_tool(function, arguments)
            print(f"   → {format_tool_preview(result)}")
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
    print(f"\n⚠️  Max turns ({MAX_TURNS}) reached. Stopping.")


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not prompt.strip():
        print("No task provided. Exiting.")
        sys.exit(1)
    agent_loop(prompt)
