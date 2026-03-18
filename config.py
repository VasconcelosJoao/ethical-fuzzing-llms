"""
Centralized configuration for ethical fuzzing campaigns.

Update PROVIDER_MODEL to change target LLMs across all campaigns.
Update K to change the number of variants per seed.
"""

PROVIDER_MODEL = {
    "gemini": "gemini-3-flash-preview",
    "openai": "gpt-5.2",
    "deepseek": "deepseek-chat",
}

K = 20  # variants per seed
