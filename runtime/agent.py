from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from law_store import LawStore, ToolInputError
from openai import OpenAI

MAX_INPUT_BYTES = 1_048_576
MAX_TOOL_OUTPUT_BYTES = 262_144

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "law_list_versions",
            "description": "List frozen versions of one legal instrument in effective-date order.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                        "description": "Instrument ID from the case JSON.",
                    }
                },
                "required": ["instrument_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "law_search_provisions",
            "description": "Search provisions within exactly one frozen history.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "history_id": {"type": "string", "minLength": 1, "maxLength": 256},
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,
                        "description": "Short whitespace-separated Korean search terms.",
                    },
                },
                "required": ["history_id", "query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "law_get_provision",
            "description": "Retrieve one exact passage by history and provision locator.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "history_id": {"type": "string", "minLength": 1, "maxLength": 256},
                    "provision_locator": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,
                    },
                },
                "required": ["history_id", "provision_locator"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "law_get_supplementary_provisions",
            "description": "Retrieve the amendment's supplementary text and articles.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "history_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,
                    }
                },
                "required": ["history_id"],
                "additionalProperties": False,
            },
        },
    },
]


def _arguments(value: str, expected: set[str]) -> dict[str, str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ToolInputError("tool arguments must be valid JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != expected:
        raise ToolInputError(f"tool arguments must contain exactly {sorted(expected)}")
    if any(not isinstance(item, str) for item in parsed.values()):
        raise ToolInputError("all tool arguments must be strings")
    return parsed


def _dispatch(store: LawStore, name: str, raw_arguments: str) -> Any:
    if name == "law_list_versions":
        args = _arguments(raw_arguments, {"instrument_id"})
        return store.list_versions(args["instrument_id"])
    if name == "law_search_provisions":
        args = _arguments(raw_arguments, {"history_id", "query"})
        return store.search_provisions(args["history_id"], args["query"])
    if name == "law_get_provision":
        args = _arguments(raw_arguments, {"history_id", "provision_locator"})
        return store.get_provision(args["history_id"], args["provision_locator"])
    if name == "law_get_supplementary_provisions":
        args = _arguments(raw_arguments, {"history_id"})
        return store.get_supplementary_provisions(args["history_id"])
    raise ToolInputError(f"unknown tool: {name}")


def _read_request() -> dict[str, Any]:
    payload = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("agent input exceeds byte limit")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("agent input must be UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != {"messages"}:
        raise ValueError("agent input must contain exactly messages")
    if not isinstance(value["messages"], list) or not value["messages"]:
        raise ValueError("messages must be a non-empty array")
    return value


def _initial_messages(raw_messages: list[Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict) or raw.get("role") not in {"system", "user"}:
            raise ValueError("initial messages may contain only system and user roles")
        content = raw.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError("initial message content must be a non-empty string")
        messages.append({"role": raw["role"], "content": content})
    if messages[-1]["role"] != "user":
        raise ValueError("the final initial message must have user role")
    return messages


def _tool_result(store: LawStore, name: str, raw_arguments: str) -> str:
    try:
        value = _dispatch(store, name, raw_arguments)
    except ToolInputError as exc:
        value = {"error": {"type": "invalid_tool_request", "message": str(exc)}}
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_TOOL_OUTPUT_BYTES:
        raise RuntimeError(f"tool output exceeds byte limit: {name}")
    return encoded


def run(*, max_model_requests: int, max_tool_calls: int) -> int:
    request = _read_request()
    messages = _initial_messages(request["messages"])
    fixture_root = os.environ.get("GOVAGENT_FIXTURE_ROOT", "/opt/law")
    store = LawStore(fixture_root)
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is required")
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "inspect-sandbox-bridge"),
        base_url=base_url,
        timeout=120.0,
        # A retry through the bridge can consume another provider generation
        # without advancing this agent's explicit request counter.
        max_retries=0,
    )

    tool_calls_used = 0
    for request_number in range(1, max_model_requests + 1):
        response = client.chat.completions.create(
            model="inspect",
            messages=messages,  # type: ignore[arg-type]
            tools=TOOLS,  # type: ignore[arg-type]
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        choice = response.choices[0]
        assistant = choice.message
        tool_calls = assistant.tool_calls or []
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": assistant.content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]
        messages.append(assistant_message)

        if not tool_calls:
            if not isinstance(assistant.content, str) or not assistant.content.strip():
                print("model returned no final submission", file=sys.stderr)
                return 24
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "model_requests": request_number,
                        "tool_calls": tool_calls_used,
                    },
                    separators=(",", ":"),
                )
            )
            return 0

        if tool_calls_used + len(tool_calls) > max_tool_calls:
            print("tool call budget exhausted", file=sys.stderr)
            return 23
        for call in tool_calls:
            tool_calls_used += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _tool_result(store, call.function.name, call.function.arguments),
                }
            )

    print("model request budget exhausted", file=sys.stderr)
    return 22


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-model-requests", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=24)
    args = parser.parse_args()
    if args.max_model_requests < 1 or args.max_tool_calls < 1:
        parser.error("budgets must be positive")
    return run(
        max_model_requests=args.max_model_requests,
        max_tool_calls=args.max_tool_calls,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"fatal agent error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
