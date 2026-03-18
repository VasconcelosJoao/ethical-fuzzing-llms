"""
Execution Module — manages API interactions with LLM providers.

Handles authentication, request submission, and response parsing
for OpenAI, DeepSeek, and Google Gemini.

Note: Rate limiting and retry logic should be added for production
campaigns (~14,160 API calls total).
"""

import os
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

load_dotenv()

client_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
client_deepseek = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)
client_gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

def run_openai(model: str, input_items, **params):
    resp = client_openai.responses.create(model=model, input=input_items, **params)
    text = getattr(resp, "output_text", None) or ""
    return {"provider": "openai", "model": model, "raw": resp, "text": text}


def run_deepseek(model: str, messages, **params):
    resp = client_deepseek.chat.completions.create(model=model, messages=messages, **params)
    text = resp.choices[0].message.content if resp.choices else ""
    return {"provider": "deepseek", "model": model, "raw": resp, "text": text}


def run_gemini(model: str, messages, **params):
    resp = client_gemini.models.generate_content(
        model=model,
        contents=messages,
        config=params
    )

    text = getattr(resp, "text", None) or ""
    return {"provider": "gemini", "model": model, "raw": resp, "text": text}
