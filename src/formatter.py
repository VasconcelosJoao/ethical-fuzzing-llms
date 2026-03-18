"""
Prompt Formatting Unit — converts ChatTurn sequences into
provider-specific API payloads.

Supports: OpenAI Responses API, DeepSeek Chat API, Google Gemini API.
Maps internal message types to appropriate roles (system/user/assistant).

References:
  - OpenAI Responses API: https://platform.openai.com/docs/api-reference
  - Gemini API: https://ai.google.dev/api/generate-content
  - Framework architecture: Section 4 of the paper
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

OpenAIRole = Literal["system", "user", "assistant"]

TYPE_TO_OPENAI_ROLE = {
    # R1 types
    "authority": "system",
    "confidential": "user",
    "benign": "user",
    "distractor": "user",
    "task": "user",
    "social_eng": "user",
    "exfiltrate": "user",
    # RF1/RF2/RF4/RT2 types (single-turn, todos user)
    "counterfactual": "user",
    "benefit_request": "user",
    "benchmark_request": "user",
    "invariance_test": "user",
    # RA2/RT1 types (multi-turn)
    "scenario": "user",
    "challenge": "user",
    "decision_request": "user",
    "explanation_request": "user",
    "assistant_decision": "assistant",  # ← CRÍTICO: resposta do modelo
}

GEMINI_ROLE_MAP = {
    "user": "user",
    "assistant": "model",
}

@dataclass
class ChatTurn:
    role: OpenAIRole
    content: str
    msg_type: str

def normalize_to_turns(messages: List[Dict[str, Any]], system_prompt: Optional[str] = None,) -> List[ChatTurn]:
    turns: List[ChatTurn] = []
    if system_prompt:
        turns.append(ChatTurn(role="system", content=system_prompt, msg_type="system_prompt"))

    for m in messages:
        if not isinstance(m, dict) or "text" not in m or "type" not in m:
            raise TypeError(f"Expected dict with 'type' and 'text', got: {m}")

        msg_type = str(m["type"])
        role = TYPE_TO_OPENAI_ROLE.get(msg_type, "user")
        turns.append(ChatTurn(role=role, content=str(m["text"]), msg_type=msg_type))

    return turns

def format_openai(model: str, turns: List[ChatTurn], **params) -> Dict[str, Any]:
    payload = {
        "model": model,
        "input": [
            {"role": t.role, "content": [{"type": "input_text", "text": t.content}]}
            for t in turns
        ],
    }
    payload.update(params)
    return payload

def format_deepseek(model: str, turns: List[ChatTurn], **params) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": t.role, "content": t.content} for t in turns],
    }
    payload.update(params)
    return payload


def format_gemini(model: str, turns, **params):
    system_parts = [t.content for t in turns if t.role == "system"]
    system_instruction = "\n".join(system_parts).strip() if system_parts else None

    contents = []
    for t in turns:
        if t.role == "system":
            continue

        gemini_role = GEMINI_ROLE_MAP.get(t.role, "user")
        contents.append({
            "role": gemini_role,
            "parts": [{"text": t.content}],
        })

    config = dict(params) if params else {}
    if system_instruction:
        config["system_instruction"] = system_instruction

    payload = {
        "model": model,
        "contents": contents,
    }
    if config:
        payload["config"] = config

    return payload

